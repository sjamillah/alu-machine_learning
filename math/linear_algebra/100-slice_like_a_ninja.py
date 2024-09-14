#!/usr/bin/env python3
"""
Slice numpy arrays
"""


def np_slice(matrix, axes=None):
    """Slices a matrix along specific axes

    Args:
        matrix (numpy.ndarray): matrix to slice
        axes (dict): dictionary where the key is an axis to slice along and
                     the value is a tuple representing the slice to make along
                     that axis

    Returns:
        numpy.ndarray: the sliced matrix
    """
    def slice_along_axis(matrix, axis, slice_tuple):
        if axis == 0:
            # Slice along the first dimension (rows)
            return [row[slice_tuple[0]:slice_tuple[1]] for row in matrix]
        elif axis == 1:
            # Slice along the second dimension (columns)
            return [row[slice_tuple[0]:slice_tuple[1]] for row in matrix]
        else:
            # For higher dimensions, more complex handling is needed
            raise NotImplementedError(
                "Slicing along this axis is not implemented."
                )

    # Apply slicing according to the axes dictionary
    for axis in sorted(axes.keys()):
        slice_tuple = axes[axis]
        matrix = slice_along_axis(matrix, axis, slice_tuple)

    return matrix
