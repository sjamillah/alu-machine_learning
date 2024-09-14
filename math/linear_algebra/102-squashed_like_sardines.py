#!/usr/bin/env python3

def cat_matrices(mat1, mat2, axis=0):
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    elif axis == 2:
        if not isinstance(mat1[0], list) or not isinstance(mat2[0], list):
            return None
        if len(mat1) != len(mat2):
            return None
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [x + y for x, y in zip(mat1, mat2)]
    else:
        return None
