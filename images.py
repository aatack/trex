import matplotlib.pyplot as plt
import numpy as np


def show(image):
    """Show the image on screen."""
    plt.imshow(image)
    plt.show()


def pixels(image, normalise=True):
    """Return a tensor representing the pixel values of an image."""
    base_array = np.array(image)
    return base_array / 256 if normalise else base_array


def greyscale(image):
    """Convert the given image to greyscale."""
    px = pixels(image)
    return 0.299 * px[:, :, 0] + 0.587 * px[:, :, 1] + 0.114 * px[:, :, 2]
