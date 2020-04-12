from __future__ import absolute_import, division, print_function

from typing import Union, Optional
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def display_images(images: Union[np.ndarray, tf.Tensor]) -> plt.Axes:
    if tf.is_tensor(images):
        images = images.numpy()
    images_to_show = np.random.randint(low=0, high=images.shape[0], size=min(9, images.shape[0]))
    images_to_show = images[images_to_show]
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    ax.axis("off")
    k = 0
    try:
        for i in range(3):
            for j in range(3):
                ax[i, j].imshow(images_to_show[k])
                k += 1
    except IndexError:
        return ax
    return ax
