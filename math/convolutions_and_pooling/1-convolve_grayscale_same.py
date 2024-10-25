#!/usr/bin/env python3
"""Performs valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs valid convolution on grayscale images

    Args:
    -images(numpy.ndarray) containing multiple
    grayscale images with shape(m, h, w):
        -m: number of images
        -h: height in pixels of the images
        -w: width in pixels of the images
    -kernel(numpy.ndarray) containing the kernel
    for the convolution with shape(kh, kw):
        -kh: height of the kernel
        -kw: width of the kernel

    Returns:
    a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    padding_width = int(kw / 2)
    padding_height = int(kh / 2)
    convolved = np.zeros((m, h, w))  # initialize output array
    # pad the images with zeros
    npad = ((0, 0), (padding_height, padding_height),
            (padding_width, padding_width))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    # the convolution operation
    for i in range(h):
        for j in range(w):
            image_patch = imagesp[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(
                np.multiply(image_patch, kernel), axis=(1, 2))
    return convolved
