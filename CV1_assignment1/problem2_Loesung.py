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
    with open(path,"rb") as f:
        data = np.load(f)
    return data

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
    # Although bayerdata.npy is a image of 2*2 np array , but it has rgb infomation.
    r,g,b = (
        np.zeros_like(bayerdata),
        np.zeros_like(bayerdata),
        np.zeros_like(bayerdata)
    )

    r[::2,1::2] = bayerdata[::2,1::2]
    g[::2,::2] = bayerdata[::2,::2]
    g[1::2,1::2] = bayerdata[1::2,1::2]
    b[1::2,::2] = bayerdata[1::2,::2]

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
    img = np.stack((r,g,b), axis =-1)
    return img


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
    '''
        rb各四分之一
        g占array的一半
        g的九宫格里面 四角还是g 这个不能算。
        rb九宫格不会再有同色
    '''

    gfilter = np.array([
        [0,1/4,0],
        [1/4,1,1/4],
        [0,1/4,0]
    ])

    rbfilter = np.array([
        [1/4,1/2,1/4],
        [1/2,1,1/2],
        [1/4,1/2,1/4]
    ])

    r = convolve(r, rbfilter, mode="mirror")
    g = convolve(g, gfilter, mode="mirror")
    b = convolve(b, rbfilter, mode="mirror")
    img = assembleimage(r,g,b)
    return img
