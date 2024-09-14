#!/usr/bin/env python3

"""
Matrix addition
"""


def add_matrices(mat1, mat2):
    # Check if the matrices have the same shape
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None


def check_shape(m1, m2):
    if isinstance(m1, list) and isinstance(m2, list):
        if len(m1) != len(m2):
            return False
        for sub1, sub2 in zip(m1, m2):
            if not check_shape(sub1, sub2):
                return False
            return True
        return True

    if not check_shape(mat1, mat2):
        return None


def add_elementwise(m1, m2):
    # Function to add matrices element-wise
    if isinstance(m1, list) and isinstance(m2, list):
        return [add_elementwise(sub1, sub2) for sub1, sub2 in zip(m1, m2)]
    else:
        return m1 + m2
    
    return add_elementwise(mat1, mat2)
