#!/usr/bin/env python3
"""
   Hue
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def change_hue(image, delta):
    """
    Changes the  hue of an image

    Param:
        - image: 3d tf.tensor
        - delta: amount of hue

    Returns:
        - altered image
    """
    return tf.image.adjust_hue(image, delta=delta)


if __name__ == "__main__":
    doggies = tfds.load("stanford_dogs", split="train", as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(change_hue(image, -0.5))
        plt.show()
