import numpy as np
from scipy.signal import convolve
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.stats import zscore

from convex_hull.offset_mask import OffsetMask
from convex_hull.region_grower import RegionGrower


class PeakIntegrator:
    @staticmethod
    def build_from_dictionary(integration_params):
        region_growth_params = {
            "distance_threshold": integration_params.pop("region_growth_distance_threshold"),
            "min_intensity": integration_params.pop("region_growth_minimum_intensity"),
            "max_size": integration_params.pop("region_growth_maximum_pixel_radius")
        }
        other_params = {
            "box_size": integration_params["peak_center_box_size"],
            "smoothing_window_size": integration_params["peak_smoothing_window_size"],
            "min_peak_pixels": integration_params["peak_minimum_pixels"],
            "min_peak_snr": integration_params["peak_minimum_signal_to_noise"],
            "outlier_threshold": integration_params["peak_pixel_outlier_threshold"]
        }
        integrator = PeakIntegrator(
            RegionGrower(**region_growth_params),
            **other_params
        )

        return integrator

    def __init__(
            self,
            region_grower,
            box_size: int = 5,
            smoothing_window_size: int = 5,
            min_peak_pixels: int = 3,
            min_peak_snr: float = 1.0,
            outlier_threshold: float = 2.0,
    ):
        """
        Integrates all peaks for an entire bank

        Parameters
        ----------
        region_grower:
            Implementation of the region growing algorithm.
        box_size:
            Size of box around estimated peak center in which to search
            for adjusted peak center. Must be odd.
        smoothing_window_size:
            Size of smoothing window for smoothing
            convolution used to find adjusted peak centers.
        min_peak_pixels:
            Minimum number of pixels in grown region needed to
            count it as a peak detection
        min_peak_snr:
            Minimum peak signal-to-noise ratio needed to count it as a peak
        outlier_threshold:
            Threshold (in # of standard deviations) for culling
            outliers in the cluster to obtain the core cluster
        """
        self.region_grower = region_grower

        assert box_size % 2 == 1, "box_size must be odd"
        self.box_size = box_size
        self.smoothing_window_size = smoothing_window_size
        self.min_peak_pixels = min_peak_pixels
        self.min_peak_snr = min_peak_snr
        self.outlier_threshold = outlier_threshold

    def integrate_peaks(
            self,
            bank_id,
            intensity,
            peak_centers,
            return_hulls=False,
            return_headers=False,
            mask=None
    ):
        """
        Integrates all peaks for the bank

        Parameters
        ----------
        bank_id:
            integer id of the bank being processed
        intensity:
            (D, H, W)-shaped array of intensities measured by the bank
        peak_centers:
            (N, 3)-shaped array of coordinates of estimated peak centers to
            compute intensity statistics for. Coordinates are in order of
            axes (D, H, W) of intensity array
        return_hulls:
            Whether to return convex hulls of peak regions for visualization
        return_headers:
            Whether to return column headers for the intensity data
        mask:
            Optional (D, H, W)-shaped boolean array indicating which pixels
            are valid

        Return
        ------
        outputs:
            Array of the peak statistics. Optionally also the peak region
            convex hulls for visualization.
        """

        # Get masks and hulls
        is_peak, peak_masks, bg_masks, peak_hulls = self._find_peak_regions(
            intensity, peak_centers, mask=mask
        )

        if return_headers:
            output_data = [[
                'bank_id', 'peak_idx', 'bg_den', 'peak_int', 'bg_int', 'sigma'
            ]]
        else:
            output_data = []

        # Use masks to compute intensity statistics
        for i_peak in range(len(peak_centers)):
            if is_peak[i_peak] and len(bg_masks[i_peak].nonzero()) > 0:
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
            else:
                bg_density, peak_intensity, peak_bg_intensity, sigma = None, None, None, None

            output_data.append([
                bank_id, i_peak, bg_density, peak_intensity, peak_bg_intensity, sigma
            ])

        if return_hulls:
            return output_data, peak_hulls
        else:
            return output_data

    def _find_peak_regions(self, intensity, peak_centers, mask=None):
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
            4-tuple of the following:

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

        # Smooth the intensity map for finding better peak centers
        smoothed_intensity = self._smooth(intensity)

        # Find regions and generate preliminary masks
        for peak_idx in range(n_peaks):
            estimated_center = tuple(map(int, peak_centers[peak_idx]))

            # Move center to local maximum *in the smoothed image*
            try:
                adjusted_center = self._local_max(smoothed_intensity, estimated_center)
            except ValueError:
                adjusted_center = None

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

            # Check if the region grew enough to be considered a peak
            if cluster_points.shape[0] < self.min_peak_pixels:
                peak_masks.append(None)
                inner_masks.append(None)
                bg_masks.append(None)
                peak_hulls.append([None]*4)
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

        return is_peak, peak_masks, bg_masks, peak_hulls

    @staticmethod
    def _calculate_statistics(intensity, peak_mask, bg_mask):
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
        print(peak_vol, bg_vol)

        total_peak_intensity = intensity[peak_indices[0], peak_indices[1], peak_indices[2]].sum()
        total_bg_intensity = intensity[bg_indices[0], bg_indices[1], bg_indices[2]].sum()
        
        print(total_peak_intensity, total_bg_intensity)

        peak_bg_intensity = peak2bg * total_bg_intensity
        peak_bg_variance = peak2bg ** 2 * total_bg_intensity

        peak_intensity = total_peak_intensity - peak_bg_intensity
        sigma = (total_peak_intensity + peak_bg_variance) ** .5

        bg_density = total_bg_intensity / bg_vol

        return bg_density, peak_intensity, peak_bg_intensity, sigma

    def _smooth(self, input_tensor):
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

    def _local_max(self, input_tensor, center_point):
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
    def _remove_outliers(data, threshold=3.0):
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

    @staticmethod
    def _find_nearest_nonzero_point(start, intensity):
        """
        Searches around a given point to find the nearest point with a
        nonzero intensity

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
        for r in range(max(intensity.shape)):
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
                            return tuple(idx)

                idx[axis] = min(start[axis] + r, intensity.shape[axis] - 1)
                for i1 in range(*ranges[0]):
                    for i2 in range(*ranges[1]):
                        idx[other_axes[0]] = i1
                        idx[other_axes[1]] = i2
                        if intensity[*idx] != 0:
                            return tuple(idx)

        raise ValueError('Invalid intensity map--all intensities are 0!')

    def _make_peak_hulls_and_masks(self, core_points, im_shape, mask=None):
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
    def _hull_mask(hull, shape, mask=None):
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
        max_vert = np.max(hull_vertices, axis=0)
        max_vert[0] = min(max_vert[0], shape[0] - 1)
        max_vert[1] = min(max_vert[1], shape[1] - 1)
        max_vert[2] = min(max_vert[2], shape[2] - 1)
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
            hull_mask &= mask[
                min_vert[0]: min_vert[0] + d_m, 
                min_vert[1]: min_vert[1] + h_m,
                min_vert[2]: min_vert[2] + w_m
            ]

        return OffsetMask(hull_mask, min_vert)

    @staticmethod
    def _expand_convex_hull(hull, scale_factor):

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
