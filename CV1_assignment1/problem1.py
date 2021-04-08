import numpy as np
import matplotlib.pyplot as plt

def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    #
    # You code here
    #

    # First imshow, second show, "Image data can not convert to float" can be avoided
    plt.imshow(np.real(img))
    plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #

    # Save the file to a binary file in NumPy ``.npy`` format.
    np.save(path,img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #

    # Load arrays from ``.npy``
    return np.load(path)


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    #
    # You code here
    #


    return np.flip(img)


def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    #
    # You code here
    #

    # Use the subplot to draw images
    plt.subplot(2, 1, 1)
    plt.imshow(np.real(img1))

    plt.subplot(2, 1, 2)
    plt.imshow(np.real(img2))
    plt.show()
