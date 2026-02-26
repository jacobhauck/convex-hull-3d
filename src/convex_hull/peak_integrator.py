import enum
from collections import namedtuple

import numpy as np
import numpy.typing as npt
from scipy.signal import convolve
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.stats import zscore

from convex_hull.offset_mask import OffsetMask
from convex_hull.region_grower import RegionGrower

IntegrationResult = namedtuple('IntegrationResult', [
    'peak_intensity',
    'peak_sigma',
    'bg_intensity',
    'bg_density',
    'peak_hull',
    'inner_hull',
    'outer_hull',
    'status',
])


class IntegrationStatus(enum.Enum):
    SUCCEEDED = 0
    NO_NONZERO_CENTER = 1
    DID_NOT_GROW = 2
    LOW_SNR = 3
    EMPTY_IMAGE = 4


class PeakIntegrator:
    def __init__(
            self,
            min_intensity: float,
            *,  # Mandatory keyword arguments
            distance_threshold: float = 3.0,
            max_size: float = 28.0,
            box_size: int = 5,
            smoothing_window_size: int = 5,
            snap_range: int = 5,
            min_peak_pixels: int = 3,
            min_peak_snr: float = 1.0,
            background_estimate: str = 'mean',
            outlier_threshold: float = 2.0,
    ):
        """
        Integrates all peaks for an entire bank

        Parameters
        ----------
        min_intensity:
            Region growth minimum intensity threshold (see RegionGrower)
        distance_threshold:
            Region growth distance threshold (see RegionGrower)
        max_size:
            Region growth maximum size (see RegionGrower)
        box_size:
            Size of box around estimated peak center in which to search
            for adjusted peak center. Must be odd.
        smoothing_window_size:
            Size of smoothing window for smoothing
            convolution used to find adjusted peak centers.
        snap_range:
            Distance from smoothed maximum point in which to search for
            a nonzero pixel to start from if the smoothed maximum point
            is zero in the original intensity map
        min_peak_pixels:
            Minimum number of pixels in grown region needed to
            count it as a peak detection
        min_peak_snr:
            Minimum peak signal-to-noise ratio needed to count it as a peak
        outlier_threshold:
            Threshold (in # of standard deviations) for culling
            outliers in the cluster to obtain the core cluster
        background_estimate:
            How to estimate the intensity of the background noise; either 
            'mean' to use the mean within the background shell or 'median' to
            use the median (excluding 0 values) within the background shell
        """
        self.region_grower = RegionGrower(
            distance_threshold=distance_threshold,
            min_intensity=min_intensity,
            max_size=max_size
        )

        assert box_size % 2 == 1, "box_size must be odd"
        self.box_size = box_size
        self.smoothing_window_size = smoothing_window_size
        self.snap_range = snap_range
        self.min_peak_pixels = min_peak_pixels
        self.min_peak_snr = min_peak_snr
        self.background_estimate = background_estimate
        self.outlier_threshold = outlier_threshold

    def integrate_peaks(
            self,
            intensity: npt.NDArray,
            peak_centers: npt.NDArray | None = None,
            mask: npt.NDArray | None = None
    ) -> list[IntegrationResult]:
        """
        Integrates all peaks for the bank

        Parameters
        ----------
        intensity:
            (D, H, W)-shaped array of intensities measured by the bank
        peak_centers:
            (N, 3)-shaped array of coordinates of estimated peak centers to
            compute intensity statistics for. Coordinates are in order of
            axes (D, H, W) of intensity array. Optionally, pass None to use the
            center of the array as the only estimated peak center.
        mask:
            Optional (D, H, W)-shaped boolean array indicating which pixels
            are valid. None (default) means all pixels are valid

        Return
        ------
        outputs:
            Length-N array of IntegrationResult named tuples containing:
            peak_intensity, peak_sigma, bg_density, bg_intensity, peak_hull,
            inner_hull, outer_hull, and status (IntegrationStatus indicating
            success/failure state of the integration)
        """

        if peak_centers is None:
            peak_centers = np.array(
                [[
                    intensity.shape[0] // 2,
                    intensity.shape[1] // 2,
                    intensity.shape[2] // 2
                ]],
                dtype=int
            )

        # Get masks and hulls
        is_peak, peak_masks, bg_masks, peak_hulls, status = self._find_peak_regions(
            intensity, peak_centers, mask=mask
        )

        output_data = []

        # Use masks to compute intensity statistics
        for i_peak in range(len(peak_centers)):
            if is_peak[i_peak] and len(bg_masks[i_peak].nonzero()[0]) > 0:
                stats = self._calculate_statistics(
                    intensity,
                    peak_masks[i_peak],
                    bg_masks[i_peak]
                )
                bg_density, peak_intensity, peak_bg_intensity, sigma = map(float, stats)

                # Discard peak if SNR is too low
                snr = peak_intensity / sigma
                if snr < self.min_peak_snr:
                    bg_density, peak_intensity, peak_bg_intensity, sigma = None, None, None, None
                    is_peak[i_peak] = False
                    peak_masks[i_peak] = None
                    bg_masks[i_peak] = None
                    peak_hulls[i_peak] = (None,) * len(peak_hulls[i_peak])
                    status[i_peak] = IntegrationStatus.LOW_SNR
            else:
                bg_density, peak_intensity, peak_bg_intensity, sigma = None, None, None, None

            output_data.append(IntegrationResult(
                peak_intensity=peak_intensity,
                peak_sigma=sigma,
                bg_intensity=peak_bg_intensity,
                bg_density=bg_density,
                peak_hull=peak_hulls[i_peak][1],
                inner_hull=peak_hulls[i_peak][2],
                outer_hull=peak_hulls[i_peak][3],
                status=status[i_peak]
            ))

        return output_data

    def _find_peak_regions(
            self,
            intensity: npt.NDArray,
            peak_centers: npt.NDArray,
            mask: npt.NDArray | None = None
    ) -> tuple[
        npt.NDArray,
        list[OffsetMask | None],
        list[OffsetMask | None],
        list[
            tuple[None, None, None, None]
            | tuple[ConvexHull, ConvexHull, ConvexHull, ConvexHull]
        ],
        list[IntegrationStatus]
    ]:
        """
        Finds peak regions based on estimated peak centers and an intensity map

        Parameters
        ----------
        intensity:
            (D, H, W)-shaped array of (integer) intensity values
        peak_centers:
            (n_peaks, 3)-shaped array of (integer) [row, col]
            coordinates of estimated peak centers
        mask:
            (D, H, W)-shaped boolean array indicating valid pixels

        Return
        ------
        outputs:
            5-tuple of the following:

            is_peak:
                (n_peaks,)-shaped array of booleans indicating whether the
                corresponding peak in peak_centers matched a peak in the intensity
                array
            peak_masks:
                list of n_peaks OffsetMasks, which
                indicate pixels that belong to the corresponding peak from
                peak_centers
            bg_masks:
                list n_peaks OffsetMasks, which indicate
                pixels that are nearby the corresponding peak from peak_centers
                but are definitely background (i.e., not contained in the inner
                region of *any* of the detected peaks)
            peak_hulls:
                list of n_peaks 4-tuples (core_hull, peak_hull, inner_hull, outer_hull)
                containing the hulls for each peak (mainly for visualization)
            status:
                list of n_peaks IntegrationStatus objects indicating current 
                integration status for each peak
        """
        # Store some basic descriptive information about inputs
        im_shape = intensity.shape
        n_peaks = len(peak_centers)

        # Initialize outputs
        is_peak = np.zeros(n_peaks, dtype=bool)
        peak_masks = []
        inner_masks = []
        bg_masks = []
        peak_hulls = []
        status = []

        # Smooth the intensity map for finding better peak centers
        smoothed_intensity = self._smooth(intensity)

        # Find regions and generate preliminary masks
        for peak_idx in range(n_peaks):
            estimated_center = tuple(map(int, peak_centers[peak_idx]))

            # Move center to local maximum *in the smoothed image*
            try:
                adjusted_center = self._local_max(smoothed_intensity, estimated_center)
                cur_status = IntegrationStatus.SUCCEEDED
            except ValueError:
                adjusted_center = None
                cur_status = IntegrationStatus.NO_NONZERO_CENTER

            # Make sure center starts from a non-zero point *in the original
            # image* because we will grow the region based on the original
            # intensity values
            try:
                if adjusted_center is not None:
                    adjusted_center = self._find_nearest_nonzero_point(adjusted_center, intensity)

                    # Grow region
                    cluster_points = self.region_grower.get_region(intensity, adjusted_center, mask=mask)
                else:
                    cluster_points = np.zeros(0)
            except ValueError:
                cluster_points = np.zeros(0)
                if cur_status == IntegrationStatus.SUCCEEDED:
                    cur_status = IntegrationStatus.EMPTY_IMAGE

            # Check if the region grew enough to be considered a peak
            if cluster_points.shape[0] < self.min_peak_pixels:
                if cur_status == IntegrationStatus.SUCCEEDED:
                    cur_status = IntegrationStatus.DID_NOT_GROW
                peak_masks.append(None)
                inner_masks.append(None)
                bg_masks.append(None)
                peak_hulls.append((None,) * 4)
                status.append(cur_status)
                continue
            else:
                is_peak[peak_idx] = True

            # Get core points of peak by removing outliers
            core_points = self._remove_outliers(
                cluster_points,
                threshold=self.outlier_threshold
            )

            # Build masks and hulls
            masks, hulls = self._make_peak_hulls_and_masks(core_points, im_shape, mask=mask)
            peak_hulls.append(hulls)
            peak_mask, inner_mask, bg_mask = masks
            peak_masks.append(peak_mask)
            inner_masks.append(inner_mask)
            bg_masks.append(bg_mask)
            status.append(cur_status)

        # Now that we have built the preliminary masks we can remove all the
        # inner region pixels from each of the background masks to avoid
        # counting any peak pixels as part of the background noise estimates
        any_inner_mask = np.zeros_like(intensity, dtype=bool)
        for inner_mask in inner_masks:
            if inner_mask is not None:
                indices = inner_mask.indices()
                any_inner_mask[indices[0], indices[1], indices[2]] |= inner_mask.mask

        # points not in *any* inner region (make this an OffsetMask so we can
        # use &= below)
        not_any_inner_mask = OffsetMask(~any_inner_mask, (0, 0, 0))

        for bg_mask in bg_masks:
            if bg_mask is not None:
                bg_mask &= not_any_inner_mask  # keep points not in any inner region

        return is_peak, peak_masks, bg_masks, peak_hulls, status

    def _calculate_statistics(
            self,
            intensity: npt.NDArray,
            peak_mask: OffsetMask,
            bg_mask: OffsetMask
    ) -> tuple[float, float, float, float]:
        """
        Calculates peak intensity statistics from a peak and background mask

        Parameters
        ----------
        intensity:
            (D, H, W) array of intensity
        peak_mask:
            OffsetMask indicating which pixels belong to the peak
        bg_mask:
            OffsetMask indicating which pixels are background near the peak

        Return
        ------
        outputs:
            4-tuple of the following:

            bg_density:
                density of intensity in the background
            peak_intensity:
                intensity of the peak alone
            peak_bg_intensity:
                intensity of the peak due to background noise
            sigma:
                estimated error in the peak intensity
        """

        peak_indices = peak_mask.nonzero()
        bg_indices = bg_mask.nonzero()

        peak_vol = len(peak_indices[0])  # number of pixels in peak
        bg_vol = len(bg_indices[0])  # number of pixels in background
        peak2bg = peak_vol / bg_vol

        total_peak_intensity = intensity[peak_indices[0], peak_indices[1], peak_indices[2]].sum()
        if self.background_estimate == 'mean':
            total_bg_intensity = intensity[bg_indices[0], bg_indices[1], bg_indices[2]].sum()
        else:
            bg_intensity = intensity[bg_indices[0], bg_indices[1], bg_indices[2]]
            bg_intensity = bg_intensity[bg_intensity > 0]
            total_bg_intensity = np.median(bg_intensity) * len(bg_indices[0])

        peak_bg_intensity = peak2bg * total_bg_intensity
        peak_bg_variance = peak2bg ** 2 * total_bg_intensity

        peak_intensity = total_peak_intensity - peak_bg_intensity
        sigma = (total_peak_intensity + peak_bg_variance) ** .5

        bg_density = total_bg_intensity / bg_vol

        return bg_density, peak_intensity, peak_bg_intensity, sigma

    def _smooth(self, input_tensor: npt.NDArray) -> npt.NDArray:
        """
        Smooths a given image by using uniform averaging kernel. Uses
        zero-padding to obtain an image with the same size.

        Parameters
        ----------
        input_tensor:
            (D, H, W)-shaped array containing the image values

        Return
        ------
        smoothed:
            (D, H, W)-shaped array with smoothed values.
        """
        # Convert to float for averaging operations
        input_tensor = input_tensor.astype(float)

        w = self.smoothing_window_size
        smoothed = convolve(
            input_tensor,
            np.ones((w, w, w)) / w**3,
            mode='same'
        )

        return smoothed

    def _local_max(
            self,
            input_tensor: npt.NDArray,
            center_point: tuple[int, ...]
    ) -> tuple[int, int, int]:
        """
        Finds the index of the pixel with the highest value in a box around
        the given center point

        Parameters
        ----------
        input_tensor:
            (D, H, W) 3D tensor
        center_point:
            [layer, row, col] coordinates of the center of the box

        Return
        ------
        max_point:
            [layer_max, row_max, col_max] coordinates of the pixel with the highest
            value in th box
        """
        d, h, w = input_tensor.shape
        l, r, c = center_point
        half = self.box_size // 2

        # Define the bounds of the window
        l_start = max(0, l - half)
        l_end = min(d, l + half + 1)
        r_start = max(0, r - half)
        r_end = min(h, r + half + 1)
        c_start = max(0, c - half)
        c_end = min(w, c + half + 1)

        # Extract the local window
        window = input_tensor[l_start:l_end, r_start:r_end, c_start:c_end]

        # Find the index of the max value in the window
        max_idx_flat = np.argmax(window)
        max_idx_3d = np.unravel_index(max_idx_flat, window.shape)

        # Map local index back to global coordinates
        global_max_idx = (
            l_start + int(max_idx_3d[0]),
            r_start + int(max_idx_3d[1]),
            c_start + int(max_idx_3d[2])
        )
        return global_max_idx

    @staticmethod
    def _remove_outliers(
            data: npt.NDArray,
            threshold: float = 3.0
    ) -> npt.NDArray:
        """
        Remove outliers from the dataset based on z-score.

        Parameters
        ----------
        data:
            Input point data of shape (n_samples, 3).
        threshold:
            Z-score threshold for outlier detection.

        Return
        ------
        filtered_data:
            Data with outliers removed.
        """
        z_scores = np.abs(zscore(data, axis=0))
        return data[(z_scores < threshold).all(axis=1)]

    def _find_nearest_nonzero_point(
            self,
            start: tuple[int, int, int],
            intensity: npt.NDArray
    ) -> tuple[int, int, int]:
        """
        Searches around a given point to find the nearest point with a
        nonzero intensity (up to self.snap_range in l^inf distance)

        Parameters
        ----------
        start: [layer, row, col] of starting point
        intensity: (D, H, W)-shaped array of intensity values

        Return
        ------
        nearest: [layer_near, row_near, col_near] coordinates of the nearest
            point to start that has a nonzero intensity
        """
        if np.all(intensity == 0):
            raise ValueError('Invalid intensity map--all intensities are 0!')

        if intensity[start[0], start[1], start[2]] != 0:
            return start

        idx = [0, 0, 0]
        for r in range(self.snap_range):
            for axis in range(3):
                other_axes = tuple(set(range(3)).difference((axis,)))
                ranges = []
                for a in other_axes:
                    ranges.append((max(0, start[a] - r), min(start[a] + r + 1, intensity.shape[a])))

                idx[axis] = max(0, start[axis] - r)
                for i1 in range(*ranges[0]):
                    for i2 in range(*ranges[1]):
                        idx[other_axes[0]] = i1
                        idx[other_axes[1]] = i2
                        if intensity[*idx] != 0:
                            return int(idx[0]), int(idx[1]), int(idx[2])

                idx[axis] = min(start[axis] + r, intensity.shape[axis] - 1)
                for i1 in range(*ranges[0]):
                    for i2 in range(*ranges[1]):
                        idx[other_axes[0]] = i1
                        idx[other_axes[1]] = i2
                        if intensity[*idx] != 0:
                            return int(idx[0]), int(idx[1]), int(idx[2])

        raise ValueError('Invalid intensity map--all intensities are 0!')

    def _make_peak_hulls_and_masks(
            self,
            core_points: npt.NDArray,
            im_shape: tuple[int, int, int],
            mask: npt.NDArray | None = None
    ) -> tuple[
        tuple[OffsetMask, OffsetMask, OffsetMask],
        tuple[ConvexHull, ConvexHull, ConvexHull, ConvexHull]
    ]:
        """
        Generate peak hulls and masks

        Parameters
        ----------
        core_points:
            (N, 3) array of coordinate vectors for points belonging to
            the core of the peak
        im_shape:
            [D, H, W] integers giving the shape of the image
        mask:
            Optional (D, H, W) mask indicating which pixels are valid

        Return
        ------
        outputs:
            2-tuple of two tuples:
            ((peak_mask, inner_mask, bg_mask), (core_hull, peak_hull, inner_hull,
              outer_hull)) giving the masks and hulls for the peak
        """
        # Adjust the core hall to make sure it's not too big or too small
        core_hull = ConvexHull(core_points)

        core_scale = 0.0
        core_hull = self._expand_convex_hull(core_hull, core_scale)

        peak_scale = 0.1
        peak_hull = self._expand_convex_hull(core_hull, peak_scale)

        inner_scale = peak_scale + 0.5
        inner_hull = self._expand_convex_hull(core_hull, inner_scale)

        outer_scale = inner_scale + 1
        outer_hull = self._expand_convex_hull(core_hull, outer_scale)

        # Generate masks
        peak_mask = self._hull_mask(peak_hull, im_shape, mask=mask)
        inner_mask = self._hull_mask(inner_hull, im_shape, mask=mask)
        outer_mask = self._hull_mask(outer_hull, im_shape, mask=mask)

        # Remove inner mask pixels to get background pixels (for now
        # ignoring the possibility of containment in a *different* peak's
        # inner region)
        bg_mask = outer_mask - inner_mask

        return ((peak_mask, inner_mask, bg_mask),
                (core_hull, peak_hull, inner_hull, outer_hull))

    @staticmethod
    def _hull_mask(
            hull: ConvexHull,
            shape: tuple[int, int, int],
            mask: npt.NDArray | None = None
    ) -> OffsetMask:
        """
        Generate an OffsetMask object with a mask that is True inside
        the given convex hull and False outside

        Parameters
        ----------
        hull:
            ConvexHull object representing the 3D convex hull.
        shape:
            tuple (D, H, W) giving the shape of the image
        mask:
            Optional mask of size (D, H, W) indicating which pixels 
            are valid

        Return
        ------
        mask:
            OffsetMask describing a mask of the convex hull
        """
        hull_vertices = hull.points[hull.vertices]
        min_vert = np.maximum(np.min(hull_vertices, axis=0), 0)
        min_vert = np.floor(min_vert).astype(int)
        max_vert = np.max(hull_vertices, axis=0)
        max_vert[0] = min(max_vert[0], shape[0] - 1)
        max_vert[1] = min(max_vert[1], shape[1] - 1)
        max_vert[2] = min(max_vert[2], shape[2] - 1)
        max_vert = np.ceil(max_vert).astype(int)
        d_m = int(max_vert[0] - min_vert[0] + 1)
        h_m = int(max_vert[1] - min_vert[1] + 1)
        w_m = int(max_vert[2] - min_vert[2] + 1)

        # Generate a Cartesian grid over the domain
        x = min_vert[0] + np.arange(d_m)
        y = min_vert[1] + np.arange(h_m)
        z = min_vert[2] + np.arange(w_m)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

        # Stack grid points into a (d, h, w, 3) array
        grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1)

        # Use Delaunay triangulation to efficiently check if points are inside the convex hull
        delaunay = Delaunay(hull.points[hull.vertices])
        hull_mask = (delaunay.find_simplex(grid_points) >= 0).astype(bool)
        if mask is not None:
            i = (
                slice(min_vert[0], min_vert[0] + d_m),
                slice(min_vert[1], min_vert[1] + h_m),
                slice(min_vert[2], min_vert[2] + w_m)
            )
            hull_mask &= mask[i[0], i[1], i[2]]

        return OffsetMask(hull_mask, min_vert)

    @staticmethod
    def _expand_convex_hull(
            hull: ConvexHull,
            scale_factor: float
    ) -> ConvexHull:

        """
        Expand a convex hull along the radial direction with respect to the mean of the points.

        Parameters
        ----------
        hull:
            ConvexHull object of the original points.
        scale_factor:
            Scaling factor for radial expansion (>0 means expansion).

        Returns
        -------
        expanded_hull:
            ConvexHull of the expanded points.
        """

        # Compute the mean of the points in the convex hull
        mean_point = np.mean(hull.points[hull.vertices], axis=0)

        # Expand each vertex along the radial direction
        delta = scale_factor * (hull.points[hull.vertices] - mean_point)
        expanded_vertex = hull.points[hull.vertices] + delta

        # Compute the new convex hull
        expanded_hull = ConvexHull(expanded_vertex)

        return expanded_hull
