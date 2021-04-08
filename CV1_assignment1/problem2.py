import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    #
    # You code here
    #

    # Load arrays from ``.npy``
    return np.load(path)

def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    #
    # You code here
    #
    
    # generate r,g,b channel
    # bayerdata.npy's shape is (512, 448)
    # How can we get the different r,g,b channel according to one channel image?
    r = np.zeros((bayerdata.shape[0],bayerdata.shape[1]),dtype=bayerdata.dtype)
    g = np.zeros((bayerdata.shape[0],bayerdata.shape[1]),dtype=bayerdata.dtype)
    b = np.zeros((bayerdata.shape[0],bayerdata.shape[1]),dtype=bayerdata.dtype)

    r[:,:] = bayerdata
    g[:,:] = bayerdata
    b[:,:] = bayerdata

    return r,g,b

def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #

    # r,g,b channel are merged to one image 
    rgb = np.dstack((r,g,b))
    return rgb


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    #
    # You code here
    #

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
    k = np.array([[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]])
    k1 = np.array([[1,1,1],[1,1,0],[1,0,0]])
    k2 = np.array([[0,1,0], [0,1,0], [0,1,0]])
    k3 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # convolve respectively
    r_result = convolve(r, k, mode='constant', cval=1.0)
    g_result = convolve(g, k2, mode='reflect')
    b_result = convolve(b, k3, mode='nearest')
    # r,g,b channel are merged to one image 
    rgb_result = np.dstack((r_result,g_result,b_result))
    return rgb_result
