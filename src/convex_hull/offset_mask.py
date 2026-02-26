"""
OffsetMask data structure for efficiently storing small masks in a large
3D image.
"""
import numpy as np
import numpy.typing as npt


class OffsetMask:
    def __init__(self, mask: npt.NDArray, offset: npt.NDArray):
        """
        Constructs an OffsetMask from the given relative mask and offset
        Parameters
        ----------
        mask: (D, H, W) Mask starting from offset
        offset: (layer, row, col) offset of mask
        """
        self.mask = mask
        self.offset = np.array(offset, dtype=int)

    def indices(self) -> tuple[slice, slice, slice]:
        """
        Return
        ------
        Indexing object into the original array giving the patch that the
        offset mask applies to
        """
        return (
            slice(self.offset[0], self.offset[0] + self.mask.shape[0]),
            slice(self.offset[1], self.offset[1] + self.mask.shape[1]),
            slice(self.offset[2], self.offset[2] + self.mask.shape[2])
        )

    def __iand__(self, other: 'OffsetMask'):
        """
        Updates this OffsetMask to be the pointwise conjunction with another
        OffsetMask

        Parameters
        ----------
        other: Another offset mask
        """
        # Probably the fastest implementation here is to create a new mask
        # array anyway, so we just fall back to __and__
        result = self & other
        self.mask = result.mask
        self.offset = result.offset

    def __ior__(self, other: 'OffsetMask'):
        """
        Updates this OffsetMask to be the pointwise disjunction with another
        OffsetMask

        Parameters
        ----------
        other: Another offset mask
        """
        result = self | other
        self.mask = result.mask
        self.offset = result.offset

    def __and__(self, other: 'OffsetMask') -> 'OffsetMask':
        """
        Returns an OffsetMask that is the pointwise conjunction with another
        OffsetMask

        Parameters
        ----------
        other: Another offset mask

        Return
        ------
        Pointwise conjunction of this offset mask with another
        """
        # Result region is intersection of two mask regions
        result_min = np.maximum(self.offset, other.offset)
        result_max = np.minimum(
            self.offset + self.mask.shape,
            other.offset + other.mask.shape
        )
        result_shape = result_max - result_min

        if np.any(result_shape < 0):
            return OffsetMask(np.zeros((0, 0, 0), dtype=bool), np.zeros(3, dtype=int))

        my_i = (
            slice(result_min[0] - self.offset[0], result_max[0] - self.offset[0]),
            slice(result_min[1] - self.offset[1], result_max[1] - self.offset[1]),
            slice(result_min[2] - self.offset[2], result_max[2] - self.offset[2])
        )
        other_i = (
            slice(result_min[0] - other.offset[0], result_max[0] - other.offset[0]),
            slice(result_min[1] - other.offset[1], result_max[1] - other.offset[1]),
            slice(result_min[2] - other.offset[2], result_max[2] - other.offset[2])
        )
        result_mask = self.mask[my_i[0], my_i[1], my_i[2]] & other.mask[other_i[0], other_i[1], other_i[2]]

        return OffsetMask(result_mask, result_min)

    def __rand__(self, other: 'OffsetMask') -> 'OffsetMask':
        """Right-sided __and__"""
        return self & other

    def __or__(self, other: 'OffsetMask') -> 'OffsetMask':
        """
        Returns an OffsetMask that is the pointwise disjunction with another
        OffsetMask

        Parameters
        ----------
        other: Another offset mask

        Return
        ------
        Pointwise disjunction of this offset mask with another
        """
        # Result region is union of two mask regions
        result_min = np.minimum(self.offset, other.offset)
        result_max = np.maximum(
            self.offset + self.mask.shape,
            other.offset + other.mask.shape
        )
        result_shape = result_max - result_min

        result_mask = np.zeros(result_shape, dtype=bool)
        my_offset_rel = self.offset - result_min
        my_i = (
            slice(my_offset_rel[0], my_offset_rel[0] + self.mask.shape[0]),
            slice(my_offset_rel[1], my_offset_rel[1] + self.mask.shape[1]),
            slice(my_offset_rel[2], my_offset_rel[2] + self.mask.shape[2])
        )
        result_mask[my_i[0], my_i[1], my_i[2]] = self.mask

        other_offset_rel = other.offset - result_min
        other_i = (
            slice(other_offset_rel[0], other_offset_rel[0] + other.mask.shape[0]),
            slice(other_offset_rel[1], other_offset_rel[1] + other.mask.shape[1]),
            slice(other_offset_rel[2], other_offset_rel[2] + other.mask.shape[2])
        )
        result_mask[other_i[0], other_i[1], other_i[2]] &= other.mask

        return OffsetMask(result_mask, result_min)

    def __ror__(self, other):
        """Right-sided version of __or__"""
        return self | other

    def __invert__(self) -> 'OffsetMask':
        """
        Returns an OffsetMask that is the pointwise negation of this one

        Return
        ------
        Pointwise conjunction of this offset mask with another
        """
        return OffsetMask(~self.mask, self.offset)

    def __sub__(self, other: 'OffsetMask') -> 'OffsetMask':
        """
        Returns an OffsetMask that contains all the points of this one and none
        of the given OffsetMask

        Parameters
        ----------
        other: Another offset mask

        Return
        ------
        Pointwise difference between this offset mask and another
        """
        # self - other = self and not other
        # Find intersection
        int_min = np.maximum(self.offset, other.offset)
        int_max = np.minimum(
            self.offset + self.mask.shape,
            other.offset + other.mask.shape
        )

        # No need to do anything if intersection is empty
        if np.any(int_max - int_min) == 0:
            return OffsetMask(self.mask.copy(), self.offset)

        # Update mask in intersection
        my_i = (
            slice(int_min[0] - self.offset[0], int_max[0] - self.offset[0]),
            slice(int_min[1] - self.offset[1], int_max[1] - self.offset[1]),
            slice(int_min[2] - self.offset[2], int_max[2] - self.offset[2])
        )
        other_i = (
            slice(int_min[0] - other.offset[0], int_max[0] - other.offset[0]),
            slice(int_min[1] - other.offset[1], int_max[1] - other.offset[1]),
            slice(int_min[2] - other.offset[2], int_max[2] - other.offset[2])
        )
        # Copy outside of intersection because ~other is True in my region
        # (set) minus other region
        result_mask = self.mask.copy()

        # and not other in the intersection
        result_mask[my_i[0], my_i[1], my_i[2]] &= ~other.mask[other_i[0], other_i[1], other_i[2]]

        return OffsetMask(result_mask, self.offset)

    def full(self, shape: tuple[int, int, int]) -> npt.NDArray:
        """
        Converts to a full mask

        Parameters
        ----------
        shape: Full array shape

        Return
        ------
        (*shape) array full mask equivalent to this OffsetMask
        """
        full_result = np.zeros(shape, dtype=bool)
        full_result[
            self.offset[0]: self.offset[0] + self.mask.shape[0],
            self.offset[1]: self.offset[1] + self.mask.shape[1],
            self.offset[2]: self.offset[2] + self.mask.shape[2]
        ] = self.mask

        return full_result

    def nonzero(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return
        ------
        tuple of 3 arrays (layer, row, col) giving the coordinates (in the
        *full* array) of the nonzero pixels specified by this mask
        """
        layer, row, col = np.nonzero(self.mask)
        layer += self.offset[0]
        row += self.offset[1]
        col += self.offset[2]
        return layer, row, col
