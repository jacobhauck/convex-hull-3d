"""
Region growing (agglomerative clustering) algorithm
"""
from math import ceil

import numpy as np
import numpy.typing as npt


def _is_valid(d, h, w, layer, row, col, mask):
    if not (0 <= layer < d and 0 <= row < h and 0 <= col < w):
        return False

    return True if mask is None else mask[layer, row, col]


class RegionGrower:
    def __init__(
            self,
            *,  # Mandatory keyword arguments
            min_intensity: float,
            distance_threshold: float = 3.0,
            max_size: float = 28.0
    ):
        """
        Parameters
        ----------
        distance_threshold: threshold for pixels to be considered neighbors
        min_intensity: minimum amount of neighboring intensity to consider
            growing the cluster from any of the neighbors of the current point
        max_size: maximum radius of the cluster
        """
        self.distance_threshold = distance_threshold
        self.min_intensity = min_intensity
        self.max_size = max_size
        self.neighbors_rel = self._get_neighbors_rel()

    def _get_neighbors_rel(self) -> list[tuple[int, int, int]]:
        """
        Precomputes neighbor relative coordinates
        :return: list of (layer, row, col) relative coordinates of neighbors
        """
        threshold_int = ceil(self.distance_threshold)
        threshold_sq = self.distance_threshold ** 2

        neighbors_rel = []
        for layer in range(-threshold_int, threshold_int + 1):
            for row in range(-threshold_int, threshold_int + 1):
                for col in range(-threshold_int, threshold_int + 1):
                    if layer**2 + row**2 + col**2 <= threshold_sq:
                        neighbors_rel.append((layer, row, col))

        return neighbors_rel

    def get_region(
            self,
            intensity: npt.NDArray,
            initial: tuple[int, int, int],
            mask: npt.NDArray | None = None
    ) -> npt.NDArray:
        """
        Gets the region by growing from the initial point

        Parameters
        ----------
        - intensity: Array of shape (D, H, W) of intensity values
        - initial: (layer, row, col) coordinates of initial point from which to grow
        - mask: optional (D, H, W) boolean array indicating which points are valid

        Return
        ------

        - cluster: (K, 3) array of coordinates of points in the cluster
        """
        initial = tuple(map(int, initial))
        visited, cluster = set(), {initial}
        grow_queue = [initial]

        while grow_queue:
            point = grow_queue.pop()
            if point in visited:
                continue

            visited.add(point)
            layer, row, col = point

            total_neighbor_intensity = 0

            neighbor_indices = []
            for neighbor_layer_rel, neighbor_row_rel, neighbor_col_rel in self.neighbors_rel:
                neighbor_layer = layer + neighbor_layer_rel
                neighbor_row = row + neighbor_row_rel
                neighbor_col = col + neighbor_col_rel

                if not _is_valid(
                        *intensity.shape,
                        neighbor_layer,
                        neighbor_row,
                        neighbor_col,
                        mask
                ):
                    continue

                neighbor_intensity = int(intensity[neighbor_layer, neighbor_row, neighbor_col])

                if neighbor_intensity > 0:
                    total_neighbor_intensity += neighbor_intensity
                    neighbor_indices.append((neighbor_layer, neighbor_row, neighbor_col))

            # Check if average intensity exceeds threshold
            n_neighbors = len(neighbor_indices)

            if n_neighbors > 0 and (total_neighbor_intensity / n_neighbors) >= self.min_intensity:
                for neighbor_point in neighbor_indices:
                    if neighbor_point in visited:
                        continue

                    if neighbor_point in cluster:
                        continue

                    neighbor_layer, neighbor_row, neighbor_col = neighbor_point
                    dist_center = (
                        (neighbor_layer - initial[0]) ** 2
                        + (neighbor_row - initial[1]) ** 2
                        + (neighbor_col - initial[2]) ** 2
                    ) ** .5
                    if dist_center < self.max_size:
                        grow_queue.append(neighbor_point)
                        cluster.add(neighbor_point)

        return np.array(list(cluster), dtype=int)
