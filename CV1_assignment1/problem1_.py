import numpy as np
import matplotlib.pyplot as plt


def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    
    plt.figure()
    plt.imshow(img)
    plt.show()
    


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """

    np.save(path, img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    img=np.load(path)
    return img



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
    print(img.shape)
    mirror_hori = np.empty_like(img)

    size = np.shape(img)
    dim=len(size)
    # image shape (height, width, channel numbers)
    # size[dim-1] is channel number
    for i in np.arange(size[dim-1]):
        # size[dim-2] is width
        for j in np.arange(size[dim-2]):
            # change width index
            mirror_hori[:,size[dim-2]-1-j,i] = img[:,j,i]

    print(np.shape(mirror_hori))
    return mirror_hori




def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    #
    # You code here
    #
    fig=plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title('Original')
    fig.add_subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('Mirror')

    plt.show()

