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
    data = np.load(path)
    # print(np.shape(data))
    # print(data)
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
    mask=np.empty((2,2,3))
    r=np.empty_like(bayerdata)
    g = np.empty_like(bayerdata)
    b = np.empty_like(bayerdata)
    dic = {0:r,1:g,2:b}
    #R
    # ?? how to know rgb matrix
    # (255, 0, 0)
    mask[:,:,0]=np.array([[0,1],
                          [0,0]])
    #G
    # (0, 255, 0)
    mask[:, :, 1] = np.array([[1, 0],
                              [0, 1]])
    #B
    # (0, 0, 255)
    mask[:, :, 2] = np.array([[0, 0],
                              [1, 0]])

    print(mask)
    print(mask.shape)

    for i in np.arange(3):
        it = np.nditer(bayerdata, flags=['multi_index'])
        for x in it: 
            
            index = it.multi_index
            # ???
            dic.get(i)[index[0], index[1]] = x * mask[: ,: ,i][index[0]%2, index[1]%2]
    print(r.dtype, np.shape(r))
    return r, g, b




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
    size=np.shape(r)
    assembleArray = np.empty((size[0],size[1],3))
    assembleArray[:,:,0]=r
    assembleArray[:, :, 1] = g
    assembleArray[:, :, 2] = b
    return assembleArray


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

    # ??? how to find proportion, how to confirm kernel
    kr=1/3
    kg=1/4
    kb=1/3
    kernel_R=kr*np.array([[1,1,1],
                             [1,1,1],
                             [1,1,1]])

    kernel_G= kg*np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]])

    kernel_B = kb*np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]])
    mode='constant'
    ip_R=  convolve(r,kernel_R, mode=mode)
    ip_G = convolve(g, kernel_G, mode=mode)
    ip_B = convolve(b, kernel_B, mode=mode)
    return assembleimage(ip_R,ip_B,ip_G)


