import multiprocessing
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve, convolve1d
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull

from convex_hull.peak_integrator import PeakIntegrator


def diameter(
        hull: ConvexHull,
        return_axes: bool = False
) -> (tuple[float, float, float]
      | tuple[float, float, float, npt.NDArray, npt.NDArray, npt.NDArray]):
    """
    Calculates the diameters of the given convex hull.

    Parameters
    ----------
    hull: a ConvexHull object
    return_axes: Whether to return the direction vectors for the
        diameters

    Return
    ------
    (d1, d2, d3), the diameters of the convex hull in order from
        largest (d1) to smallest (d3). Optionally also (v1, v2, v3), unit
        vectors giving the directions of each diameter.
    """
    # Calculate matrix of pairwise distances
    verts = hull.points[hull.vertices]
    dists = np.linalg.norm(verts[None, :, :] - verts[:, None, :], axis=-1)

    # Find the pair of points with maximum distance; convexity => this is the
    # diameter
    indices = np.argmax(dists, axis=0)
    maxes = np.take_along_axis(dists, indices[None], axis=0)
    j_big = np.argmax(maxes)
    i_big = indices[j_big]

    # Project points onto the plane orthogonal to first diameter
    centered = verts - np.mean(verts, axis=1)[:, None]
    big_dir = verts[j_big] - verts[i_big]
    big_dir /= np.linalg.norm(big_dir)
    projected = centered - big_dir[None] * np.einsum('ni,i->n', centered, big_dir)[:, None]

    # Repeat diameter calculation in projected plane
    pdists = np.linalg.norm(projected[None] - projected[:, None, :], axis=-1)
    indices = np.argmax(pdists, axis=0)
    mins = np.take_along_axis(pdists, indices[None], axis=0)
    j_med = np.argmax(mins)
    i_med = indices[j_med]

    # Project onto the line orthogonal to the first two diameters
    med_dir = projected[j_med] - projected[i_med]
    med_dir /= np.linalg.norm(med_dir)
    projected2 = projected - med_dir[None] * np.einsum('ni,i->n', projected, med_dir)[:, None]

    # Repeat diameter calculation on projected line
    pdists2 = np.linalg.norm(projected2[None] - projected2[:, None, :], axis=-1)
    indices = np.argmax(pdists2, axis=0)
    mins = np.take_along_axis(pdists2, indices[None], axis=0)
    j_small = np.argmax(mins)
    i_small = indices[j_small]

    small_dir = projected2[j_small] - projected2[i_small]
    
    if not return_axes:
        return dists[i_big, j_big], pdists[i_med, j_med], pdists2[i_small, j_small]
    else:
        return (
            dists[i_big, j_big], pdists[i_med, j_med], pdists2[i_small, j_small],
            big_dir, med_dir, small_dir
        )


def find_first_nonzero_volume(
        peak_integrator: PeakIntegrator,
        events: npt.NDArray,
        detector_mask: npt.NDArray
) -> float:
    """
    Finds the largest min_intensity threshold parameter of the given
    peak_integrator that returns a nonzero integration for the given event
    histogram.

    Parameters
    ----------
    peak_integrator: PeakIntegrator to use for integration
    events: (D, H, W) histogram of detector events
    detector_mask: (D, H, W) boolean-valued array indicating valid
        detector pixels

    Return
    ------
    Largest min_intensity threshold parameter of peak_integrator that
        provides the first nonzero integration
    """
    original_min_intensity = peak_integrator.region_grower.min_intensity
    min_intensity = 0
    max_intensity = np.max(events)
    cur_min_intensity = max_intensity / 2
    while max_intensity - cur_min_intensity > 0.1:
        peak_integrator.region_grower.min_intensity = min_intensity
        integration_result = peak_integrator.integrate_peaks(events, mask=detector_mask)[0]
        
        if integration_result.peak_intensity is None or integration_result.peak_hull.volume == 0:
            max_intensity = cur_min_intensity
            cur_min_intensity = (min_intensity + cur_min_intensity) / 2.0
        else:
            min_intensity = cur_min_intensity
            cur_min_intensity = (max_intensity + cur_min_intensity) / 2.0

    # Reset original min_intensity
    peak_integrator.region_grower.min_intensity = original_min_intensity

    return cur_min_intensity


class BaseThreshold:
    """Base class for thresholding algorithms"""
    def get_threshold(self, events: npt.NDArray, detector_mask: npt.NDArray) -> float:
        """Calculates the threshold"""
        raise NotImplemented

    def close(self):
        """Cleans up state, if necessary"""
        pass


class ZScoreThreshold(BaseThreshold):
    def __init__(self, n_sigma: float, min_threshold: float):
        """
        Sets the min_intensity threshold equal to the value of the events
        histogram at a specified z score

        Parameters
        ----------
        n_sigma: Number of standard deviations (i.e., z score) at which to set
            the threshold
        min_threshold: Minimum threshold
        """
        self.n_sigma = n_sigma
        self.min_threshold = min_threshold

    def get_threshold(self, events: npt.NDArray, detector_mask: npt.NDArray) -> float:
        m = float(np.mean(events[detector_mask]))
        s = float(np.std(events[detector_mask]))
        min_intensity = max(
            self.min_threshold,
            m + s * self.n_sigma
        )
        return min_intensity


class MaxRelativeThreshold(BaseThreshold):
    def __init__(self, frac_max, min_threshold):
        """
        Sets the min_intensity threshold to a specified fraction of the maximum
        intensity in the event histogram.

        Parameters
        ----------
        frac_max: Fraction of the maximum intensity
        min_threshold: Minimum allowable threshold
        """
        self.frac_max = frac_max
        self.min_threshold = min_threshold

    def get_threshold(self, events: npt.NDArray, detector_mask: npt.NDArray) -> float:
        m = np.max(events[detector_mask])
        return max(self.min_threshold, self.frac_max * m)


class MaxVarianceThreshold(BaseThreshold):
    def __init__(self, num_levels, min_threshold):
        """
        Sets the min_intensity threshold that maximizes the variance of the
        events above the threshold

        Parameters
        ----------
        num_levels: Number of levels to use in line search for the
            optimal threshold
        min_threshold: Minimum allowable threshold
        """
        self.num_levels = num_levels
        self.min_threshold = min_threshold

    def get_threshold(self, events: npt.NDArray, detector_mask: npt.NDArray) -> float:
        x = events[detector_mask]
        levels = np.linspace(0, 1, self.num_levels)
        levels = (np.exp(levels) - 1) / (np.exp(1) - 1) * (np.max(x) - np.min(x)) + np.min(x)
        v = []
        for l in levels:
            v.append(np.var(x[x >= l]))
        return max(self.min_threshold, levels[np.argmax(v)])


class DensityBasedThreshold(BaseThreshold):
    def __init__(self, num_levels, cutoff, min_threshold):
        """
        Sets the min_intensity threshold at the first point where the PDF of the
        event counts passes below a given cutoff

        Parameters
        ----------
        num_levels: Number of levels to use in line search for optimal
            threshold
        cutoff: PDF cutoff
        min_threshold: Minimum allowable threshold
        """
        self.num_levels = num_levels
        self.cutoff = cutoff
        self.min_threshold = min_threshold

    def get_threshold(self, events: npt.NDArray, detector_mask: npt.NDArray) -> float:
        x = events[detector_mask]
        levels = np.linspace(0, 1, self.num_levels)
        min_x, max_x = np.min(x), np.max(x)
        if max_x - min_x < 10:
            return 100
        levels = (np.exp(levels) - 1) / (np.exp(1) - 1) * (np.max(x) - np.min(x)) + np.min(x)
        counts = []
        for l in levels:
            counts.append(len(x[x >= l]))
        counts = np.array(counts)
        counts_smooth = np.array([np.mean(counts[i - 10 : i + 10]) for i in range(10, len(counts) - 10)])
        delta = (counts_smooth[1:] - counts_smooth[:-1]) / (levels[11:-10] - levels[10:-11])
        # noinspection PyTypeChecker
        index = np.nonzero(-delta < self.cutoff)[0][0]
        return max(self.min_threshold, levels[index])


class QuantileThreshold(BaseThreshold):
    def __init__(self, quantile, smoothing_kernel_size, min_threshold):
        """
        Sets the min_intensity threshold to a specified quantile of the event
        count distribution

        Parameters
        ----------
        quantile: Specified quantile (0 < quantile < 1)
        smoothing_kernel_size: Size of smoothing kernel to apply to
            events before computing quantile
        min_threshold: Minimum allowable threshold
        """
        self.quantile = quantile
        self.min_threshold = min_threshold
        shape = (smoothing_kernel_size,) * 3
        smoothing_kernel = np.ones(shape)
        smoothing_kernel /= smoothing_kernel.size
        self.kernel = smoothing_kernel

    def get_threshold(self, events: npt.NDArray, detector_mask: npt.NDArray) -> float:
        smooth_events: npt.NDArray = convolve(
            events.astype(float),
            self.kernel.astype(float),
            mode="constant",
            cval=0.0
        )
        min_intensity = np.quantile(smooth_events.flatten(), self.quantile)
        return max(self.min_threshold, min_intensity)


def shape_scan_step(
        integrator: PeakIntegrator,
        events: npt.NDArray,
        detector_mask: npt.NDArray,
        min_intensity: float
) -> tuple[float, float, float]:
    """
    Does one step of the shape scan algorithm

    Parameters
    ----------
    integrator: PeakIntegrator to use for integration
    events: (D, H, W) array of events
    detector_mask: (D, H, W) detector mask array
    min_intensity: threshold to use for the shape scan step

    Return
    ------
    The three diameters of the fitted convex hull
    """
    original = integrator.region_grower.min_intensity
    integrator.region_grower.min_intensity = min_intensity
    integration_result = integrator.integrate_peaks(events, mask=detector_mask)[0]
    integrator.region_grower.min_intensity = original

    if integration_result.peak_hull is None:
        return 0.0, 0.0, 0.0
    else:
        return diameter(integration_result.peak_hull)


class ShapeScanThreshold(BaseThreshold):
    def __init__(
            self,
            *,  # Mandatory keyword arguments
            start_intensity: float,
            scan_decay: float,
            integrator: PeakIntegrator,
            min_threshold: float,
            num_processes: int = 0
    ):
        """
        Sets the min_intensity threshold to the value that maximizes the
        difference between the first and second diameters of the fitted peak

        Parameters
        ----------
        start_intensity: Largest min_intensity threshold to try
        scan_decay: Amount by which to reduce the threshold on each scan step
        integrator: PeakIntegrator to use for integration
        min_threshold: Minimum allowable threshold
        num_processes: Number of parallel processes to use to perform the scan
            integrations. Set to 0 (default) to do scan without multiprocessing.
            If you use multiprocessing, remember to call close() to close the
            process pool when you are done with this object.
        """
        self.start_intensity = start_intensity
        self.scan_decay = scan_decay
        self.min_threshold = min_threshold
        self.num_processes = num_processes
        self.integrator = integrator
        if num_processes > 0:
            self.pool = multiprocessing.Pool(num_processes)
        else:
            self.pool = None
        
    def get_threshold(self, events: npt.NDArray, detector_mask: npt.NDArray) -> float:
        if self.start_intensity < 0:
            min_intensity = np.max(events)
        else:
            min_intensity = self.start_intensity

        scan_min_intensities = []
        while min_intensity > self.min_threshold:
            scan_min_intensities.append(min_intensity)
            min_intensity *= self.scan_decay

        if self.num_processes == 0: 
            big_sizes, med_sizes, small_sizes = [], [], []
            for min_intensity in scan_min_intensities:
                big, med, small = shape_scan_step(
                    self.integrator, events, detector_mask, min_intensity
                )
                big_sizes.append(big)
                med_sizes.append(med)
                small_sizes.append(small)
        else:
            if len(scan_min_intensities) > 0:
                returns = self.pool.map(
                    partial(shape_scan_step, self.integrator, events, detector_mask),
                    scan_min_intensities
                )
                big_sizes, med_sizes, small_sizes = zip(*returns)
            else:
                big_sizes, med_sizes, small_sizes = [], [], []

        if len(big_sizes) > 0:
            size_gaps = np.array(big_sizes) - np.array(med_sizes)
            min_intensity = scan_min_intensities[np.argmax(size_gaps)]
        else:
            min_intensity = self.min_threshold

        return min_intensity

 
def snr_scan_step(
        integrator: PeakIntegrator,
        events: npt.NDArray,
        detector_mask: npt.NDArray,
        min_intensity: float
) -> float | None:
    """
    Performs one step of the SNR scan algorithm

    Parameters
    ----------
    integrator: PeakIntegrator to use for integration
    events: (D, H, W) array of events
    detector_mask: (D, H, W) array of detector mask
    min_intensity: threshold to use

    Return
    ------
    SNR for the current peak integration
    """
    original = integrator.region_grower.min_intensity
    integrator.region_grower.min_intensity = min_intensity
    integration_result = integrator.integrate_peaks(events, mask=detector_mask)[0]
    integrator.region_grower.min_intensity = original

    if integration_result.peak_intensity is None or integration_result.peak_sigma == 0:
        return None
    else:
        return integration_result.peak_intensity / integration_result.peak_sigma



class SNRScanThreshold(BaseThreshold):
    def __init__(
            self,
            *,  # Mandatory keyword arguments
            start_intensity: float,
            scan_steps: int,
            integrator: PeakIntegrator,
            min_threshold: float,
            num_processes: int = 0
    ):
        """
        Sets the min_intensity threshold to the largest value that is a local
        maximizer of the signal-to-noise ratio (SNR) of the integrate peak

        Parameters
        ----------
        start_intensity: Smallest intensity to use in the scan for local maxima
        scan_steps: Number of steps in the scan
        integrator: PeakIntegrator to use for integration
        min_threshold: Minimum allowable threshold
        num_processes: Number of parallel processes to use to perform the scan
            integrations. Set to 0 (default) to do scan without multiprocessing.
            If you use multiprocessing, remember to call close() to close the
            process pool when you are done with this object.
        """
        self.start_intensity = start_intensity
        self.scan_steps = scan_steps
        self.integrator = integrator
        self.min_threshold = min_threshold
        self.num_processes = num_processes
        self.scan_min_intensities = None
        self.scan_snr = None
        self.snr_smooth = None
        if num_processes > 0:
            self.pool = multiprocessing.Pool(num_processes)
        else:
            self.pool = None

    def close(self):
        self.pool.close()

    def get_threshold(self, events, detector_mask):
        max_intensity = find_first_nonzero_volume(self.integrator, events, detector_mask)
        self.scan_min_intensities = np.linspace(
            self.start_intensity,
            max_intensity,
            self.scan_steps
        )

        if self.num_processes == 0:
            scan_snr = []
            valid_min_intensities = []
            for min_intensity in self.scan_min_intensities:
                snr = snr_scan_step(
                    self.integrator, events, detector_mask, min_intensity
                )
                if snr is not None:
                   valid_min_intensities.append(min_intensity)
                   scan_snr.append(snr)

            scan_snr = np.array(scan_snr)
            scan_min_intensities = np.array(valid_min_intensities)
        else:
            returns = self.pool.map(
                partial(snr_scan_step, self.integrator, events, detector_mask),
                self.scan_min_intensities
            )
            scan_snr = np.array([r for r in returns if r is not None])
            scan_min_intensities = np.array([
                i for r, i in zip(returns, self.scan_min_intensities)
                if r is not None
            ])

        scan_snr = scan_snr[:-1]  # Drop last SNR
        scan_min_intensities = scan_min_intensities[:-1]
        self.scan_snr = scan_snr
        self.scan_min_intensities = scan_min_intensities

        if len(scan_snr) == 0:
            min_intensity = self.min_threshold
        else:
            kernel = np.exp(-(np.arange(9) - 4)**2/2/1.6**2)
            kernel /= kernel.sum()
            snr_smooth: npt.NDArray = convolve1d(scan_snr, kernel)
            self.snr_smooth = snr_smooth
            snr_peaks, _ = find_peaks(snr_smooth)
            m = 0.9 * max(scan_min_intensities)
            # noinspection PyUnresolvedReferences
            snr_peaks = snr_peaks[scan_min_intensities[snr_peaks] < m]
            if len(snr_peaks) == 0:
                min_intensity = self.min_threshold
            else:
                i_best = np.argmax(scan_min_intensities[snr_peaks])
                min_intensity = scan_min_intensities[snr_peaks[i_best]]

        return min_intensity
