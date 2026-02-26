import multiprocessing
from functools import partial

import numpy as np
from scipy.ndimage import convolve, convolve1d
from scipy.signal import find_peaks

import convex_hull


def diameter(hull, return_axes=False):
    verts = hull.points[hull.vertices]
    dists = np.linalg.norm(verts[None, :, :] - verts[:, None, :], axis=-1)
    inds = np.argmax(dists, axis=0)
    maxes = np.take_along_axis(dists, inds[None], axis=0)
    j_big = np.argmax(maxes)
    i_big = inds[j_big]
   
    centered = verts - np.mean(verts, axis=1)[:, None]
    big_dir = verts[j_big] - verts[i_big]
    big_dir /= np.linalg.norm(big_dir)
    projected = centered - big_dir[None] * np.einsum('ni,i->n', centered, big_dir)[:, None]
    pdists = np.linalg.norm(projected[None] - projected[:, None, :], axis=-1) 
    inds = np.argmax(pdists, axis=0)
    mins = np.take_along_axis(pdists, inds[None], axis=0)
    j_med = np.argmax(mins)
    i_med = inds[j_med]
    
    med_dir = projected[j_med] - projected[i_med]
    med_dir /= np.linalg.norm(med_dir)
    projected2 = projected - med_dir[None] * np.einsum('ni,i->n', projected, med_dir)[:, None]
    pdists2 = np.linalg.norm(projected2[None] - projected2[:, None, :], axis=-1) 
    inds = np.argmax(pdists2, axis=0)
    mins = np.take_along_axis(pdists2, inds[None], axis=0)
    j_small = np.argmax(mins)
    i_small = inds[j_small]

    small_dir = projected2[j_small] - projected2[i_small]
    
    if not return_axes:
        return dists[i_big, j_big], pdists[i_med, j_med], pdists2[i_small, j_small]
    else:
        return big_dir, med_dir, small_dir, dists[i_big, j_big], pdists[i_med, j_med], pdists2[i_small, j_small]


def find_first_nonzero_volume(ch_args, events, detector_mask):
    min_intensity = 0
    max_intensity = np.max(events)
    cur_min_intensity = max_intensity / 2
    while max_intensity - cur_min_intensity > 0.1:
        integration_result = convex_hull.integrate_peak(
            events,
            detector_mask,
            distance_threshold=ch_args.distance_threshold,
            min_intensity=cur_min_intensity,
            max_size=ch_args.max_size,
            box_search_size=3,
            smoothing_window_size=3,
            snap_range=8,
            min_peak_pixels=ch_args.min_peak_pixels,
            min_peak_snr=1.0,
            outlier_threshold=2.0,
            return_hulls=True,
            background_estimate=ch_args.background_estimate
        )
        
        if integration_result.peak_intensity is None or integration_result.peak_hull.volume == 0:
            max_intensity = cur_min_intensity
            cur_min_intensity = (min_intensity + cur_min_intensity) / 2.0
        else:
            min_intensity = cur_min_intensity
            cur_min_intensity = (max_intensity + cur_min_intensity) / 2.0
        
    return cur_min_intensity


class BaseThreshold:
    def get_threshold(self, events, detector_mask):
        raise NotImplemented

    def close(self):
        pass


class ZScoreThreshold(BaseThreshold):
    def __init__(self, n_sigma, min_threshold):
        self.n_sigma = n_sigma
        self.min_threshold = min_threshold

    def get_threshold(self, events, detector_mask):
        m = np.mean(events[detector_mask])
        s = np.std(events[detector_mask])
        min_intensity = max(
            self.min_threshold,
            m + s * self.n_sigma
        )
        return min_intensity


class MaxRelativeThreshold(BaseThreshold):
    def __init__(self, frac_max, min_threshold):
        self.frac_max = frac_max
        self.min_threshold = min_threshold

    def get_threshold(self, events, detector_mask):
        m = np.max(events[detector_mask])
        return max(self.min_threshold, self.frac_max * m)


class MaxVarianceThreshold(BaseThreshold):
    def __init__(self, num_levels, min_threshold):
        self.num_levels = num_levels
        self.min_threshold = min_threshold

    def get_threshold(self, events, detector_mask):
        x = events[detector_mask]
        levels = np.linspace(0, 1, self.num_levels)
        levels = (np.exp(levels) - 1) / (np.exp(1) - 1) * (np.max(x) - np.min(x)) + np.min(x)
        v = []
        for l in levels:
            v.append(np.var(x[x >= l]))
        m = (np.max(v) + np.min(v)) / 2
        where_above = np.nonzero(v >= m)[0]
        return max(self.min_threshold, levels[np.argmax(v)])


class DensityBasedThreshold(BaseThreshold):
    def __init__(self, num_levels, cutoff, min_threshold):
        self.num_levels = num_levels
        self.cutoff = cutoff
        self.min_threshold = min_threshold

    def get_threshold(self, events, detector_mask):
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
        index = np.nonzero(-delta < self.cutoff)[0][0]
        return max(self.min_threshold, levels[index])


class QuantileThreshold(BaseThreshold):
    def __init__(self, quantile, smoothing_kernel_size, min_threshold):
        self.quantile = quantile
        self.min_threshold = min_threshold
        shape = (smoothing_kernel_size,) * 3
        smoothing_kernel = np.ones(shape)
        smoothing_kernel /= smoothing_kernel.size
        self.kernel = smoothing_kernel

    def get_threshold(self, events, detector_mask):
        smooth_events = convolve(
            events.astype(float),
            self.kernel.astype(float),
            mode="constant",
            cval=0.0
        )
        min_intensity = np.quantile(smooth_events.flatten(), self.quantile)
        return max(self.min_threshold, min_intensity)


def shape_scan_step(ch_args, events, detector_mask, min_intensity):
    integration_result = convex_hull.integrate_peak(
        events,
        detector_mask,
        distance_threshold=ch_args.distance_threshold,
        min_intensity=min_intensity,
        max_size=ch_args.max_size,
        box_search_size=3,
        smoothing_window_size=3,
        snap_range=8,
        min_peak_pixels=ch_args.min_peak_pixels,
        min_peak_snr=1.0,
        outlier_threshold=2.0,
        return_hulls=True
    )
    
    if integration_result.peak_hull is None:
        return 0.0, 0.0, 0.0
    else:
        return diameter(integration_result.peak_hull)


class ShapeScanThreshold(BaseThreshold):
    def __init__(self, start_intensity, scan_decay, ch_args, min_threshold, num_processes):
        self.start_intensity = start_intensity
        self.scan_decay = scan_decay
        self.min_threshold = min_threshold
        self.num_processes = num_processes
        self.ch_args = ch_args
        if num_processes > 0:
            self.pool = multiprocessing.Pool(num_processes)
        else:
            self.pool = None
        
    def get_threshold(self, events, detector_mask):
        if self.start_intensity < 0:
            min_intensity = np.max(events)
        else:
            min_intensity = self.start_intensity

        scan_min_intensities = []
        while min_intensity > self.min_threshold:
            scan_min_intensities.append(min_intensity)
            min_intensity *= self.scan_decay
        print(scan_min_intensities)

        if self.num_processes == 0: 
            big_sizes, med_sizes, small_sizes = [], [], []
            for min_intensity in scan_min_intensities:
                big, med, small = shape_scan_step(self.ch_args, events, detector_mask, min_intensity)
                big_sizes.append(big)
                med_sizes.append(med)
                small_sizes.append(small)
        else:
            if len(scan_min_intensities) > 0:
                returns = self.pool.map(
                    partial(shape_scan_step, self.ch_args, events, detector_mask),
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


def intensity_scan_step(ch_args, events, detector_mask, min_intensity):
    integration_result = convex_hull.integrate_peak(
        events,
        detector_mask,
        distance_threshold=ch_args.distance_threshold,
        min_intensity=min_intensity,
        max_size=ch_args.max_size,
        box_search_size=3,
        smoothing_window_size=3,
        snap_range=8,
        min_peak_pixels=ch_args.min_peak_pixels,
        min_peak_snr=1.0,
        outlier_threshold=2.0,
        return_hulls=True
    )

    if integration_result.peak_intensity is None:
        return 0.0
    else:
        return integration_result.peak_intensity


class IntensityScanV1Threshold(BaseThreshold):
    def __init__(
            self,
            start_intensity,
            scan_decay,
            di_threshold,
            ch_args,
            min_threshold,
            num_processes
    ):
        self.start_intensity = start_intensity
        self.scan_decay = scan_decay
        self.ch_args = ch_args
        self.di_threshold = di_threshold
        self.min_threshold = min_threshold
        self.num_processes = num_processes
        if self.num_processes > 0:
            self.pool = multiprocessing.Pool(num_processes) 
        else:
            self.pool = None

    def get_threshold(self, events, detector_mask):
        min_intensity = np.max(events)
        min_intensity = self.start_intensity

        scan_min_intensities = []
        while min_intensity < max_intensity:
            scan_min_intensities.append(min_intensity)
            min_intensity /= self.scan_decay
        scan_min_intensities = np.array(scan_min_intensities)

        if self.num_processes == 0: 
            scan_intensities = []
            for min_intensity in scan_min_intensities:
                cur_intensity = intensity_scan_step(self.ch_args, events, detector_mask, min_intensity)
                scan_intensities.append(cur_intensity)
            scan_intensities = np.array(scan_intensities)
        else:
            if len(scan_min_intensities) > 0:
                returns = self.pool.map(
                    partial(intensity_scan_step, self.ch_args, events, detector_mask),
                    scan_min_intensities
                )
                scan_intensities = np.array([r for r in returns])
            else:
                scan_intensities = []

        if len(scan_intensities) <= 3:
            min_intensity = self.min_threshold
        else:
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            last_zero = len(scan_intensities) - 1
            while last_zero >= 1 and scan_intensities[last_zero] == 0:
                last_zero -= 1
            scan_intensities = scan_intensities[:last_zero]
            scan_min_intensities = scan_min_intensities[:last_zero]
            smooth = convolve1d(scan_intensities, kernel)
            with np.errstate(all='ignore'):
                di = (smooth[2:] - smooth[:-2]) / (scan_min_intensities[2:] - scan_min_intensities[:-2])
        
            for j in range(len(di) - 2, -1, -1):
                if di[j+1] == 0:
                    continue
                change = max(abs(di[j]) / abs(di[j+1]), abs(di[j+1]) / abs(di[j]))
                if change > self.di_threshold:
                    min_intensity = scan_min_intensities[j+2]
                    break
            else:
                min_intensity = self.min_threshold

        return min_intensity


class IntensityScanV2Threshold(BaseThreshold):
    def __init__(
            self,
            start_intensity,
            scan_decay,
            di_threshold,
            ch_args,
            min_threshold,
            num_processes
    ):
        self.start_intensity = start_intensity
        self.scan_decay = scan_decay
        self.di_threshold = di_threshold
        self.ch_args = ch_args
        self.min_threshold = min_threshold
        self.num_processes = num_processes
        if self.num_processes > 0:
            self.pool = multiprocessing.Pool(num_processes) 
        else:
            self.pool = None

    def get_threshold(self, events, detector_mask):
        min_intensity = np.max(events)
        min_intensity = self.start_intensity

        scan_min_intensities = []
        while min_intensity < max_intensity:
            scan_min_intensities.append(min_intensity)
            min_intensity /= self.scan_decay
        scan_min_intensities = np.array(scan_min_intensities)

        if self.num_processes == 0: 
            scan_intensities = []
            for min_intensity in scan_min_intensities:
                cur_intensity = intensity_scan_step(self.ch_args, events, detector_mask, min_intensity)
                scan_intensities.append(cur_intensity)
            scan_intensities = np.array(scan_intensities)
        else:
            if len(scan_min_intensities) > 0:
                returns = self.pool.map(
                    partial(intensity_scan_step, self.ch_args, events, detector_mask),
                    scan_min_intensities
                )
                scan_intensities = np.array([r for r in returns])
            else:
                scan_intensities = []

        if len(scan_intensities) <= 3:
            min_intensity = self.min_threshold
        else:
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            smooth = convolve1d(scan_intensities, kernel)
            with np.errstate(all='ignore'):
                di = (smooth[2:] - smooth[:-2]) / (scan_min_intensities[2:] - scan_min_intensities[:-2])

            for j in range(len(di) - 2, -1, -1):
                if di[j+1] == 0:
                    continue
                change = abs(di[j+1] - di[j]) / abs(di[j+1])
                if change > self.di_threshold:
                    min_intensity = scan_min_intensities[j+2]
                    break
            else:
                min_intensity = self.min_threshold

        return min_intensity
 
 
def snr_scan_step(ch_args, events, detector_mask, min_intensity):
    integration_result = convex_hull.integrate_peak(
        events,
        detector_mask,
        distance_threshold=ch_args.distance_threshold,
        min_intensity=min_intensity,
        max_size=ch_args.max_size,
        box_search_size=3,
        smoothing_window_size=3,
        snap_range=8,
        min_peak_pixels=ch_args.min_peak_pixels,
        min_peak_snr=1.0,
        background_estimate=ch_args.background_estimate,
        outlier_threshold=2.0,
        return_hulls=True
    )

    if integration_result.peak_intensity is None or integration_result.peak_sigma == 0:
        return None
    else:
        return integration_result.peak_intensity / integration_result.peak_sigma



class SNRScanThreshold(BaseThreshold):
    def __init__(
            self,
            start_intensity,
            scan_steps,
            ch_args,
            min_threshold,
            num_processes
    ):
        self.start_intensity = start_intensity
        self.scan_steps = scan_steps
        self.ch_args = ch_args
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
        max_intensity = find_first_nonzero_volume(self.ch_args, events, detector_mask)
        self.scan_min_intensities = np.linspace(
            self.start_intensity,
            max_intensity,
            self.scan_steps
        )

        if self.num_processes == 0:
            scan_snr = []
            valid_min_intensities = []
            for min_intensity in self.scan_min_intensities:
                snr = snr_scan_step(self.ch_args, events, detector_mask, min_intensity)
                if snr is not None:
                   valid_min_intensities.append(min_intensity)
                   scan_snr.append(snr)

            scan_snr = np.array(scan_snr)
            scan_min_intensities = np.array(valid_min_intensities)
        else:
            returns = self.pool.map(
                partial(snr_scan_step, self.ch_args, events, detector_mask),
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
            snr_smooth = convolve1d(scan_snr, kernel)
            self.snr_smooth = snr_smooth
            snr_peaks, _ = find_peaks(snr_smooth)
            m = 0.9 * max(scan_min_intensities)
            snr_peaks = snr_peaks[scan_min_intensities[snr_peaks] < m]
            if len(snr_peaks) == 0:
                min_intensity = self.min_threshold
            else:
                i_best = np.argmax(scan_min_intensities[snr_peaks])
                min_intensity = scan_min_intensities[snr_peaks[i_best]]

        return min_intensity
 
