import math
import numpy as np
from scipy import ndimage


def gauss2d(sigma, fsize):
    """
    Args:
      sigma: width of the Gaussian filter
      fsize: dimensions of the filter

    Returns:
      g: *normalized* Gaussian filter
    """

    #
    # You code here
    #


def createfilters():
    """
    Returns:
      fx, fy: filters as described in the problem assignment
    """

    #
    # You code here
    #
    # ???
    y_Gaussian = 0.5 * np.array([0.5395, 1, 0.5395])
    y_Gaussian = np.reshape(y_Gaussian, (3, 1))
    D_x = 0.5 * np.array([1, 0, -1])
    D_x = np.reshape(D_x, (1, 3))
    DF_x = y_Gaussian * D_x

    x_Gaussian = 0.5 * np.array([0.5395, 1, 0.5395])
    x_Gaussian = np.reshape(x_Gaussian, (1, 3))
    D_y = 0.5 * np.array([1, 0, -1])
    D_y = np.reshape(D_y, (3, 1))
    DF_y = x_Gaussian * D_y

    return DF_x, DF_y


def filterimage(I, fx, fy):
    """ Filter the image with the filters fx, fy.
    You may use the ndimage.convolve scipy-function.

    Args:
      I: a (H,W) numpy array storing image data
      fx, fy: filters

    Returns:
      Ix, Iy: images filtereFor d by fx and fy respectively
    """

    #
    # You code here
    #

    # no partial derivatives???
    Ix = ndimage.convolve(I, fx)
    Iy = ndimage.convolve(I, fy)
    return Ix, Iy


def detectedges(Ix, Iy, thr):
    """ Detects edges by applying a threshold on the image gradient magnitude.

    Args:
      Ix, Iy: filtered images
      thr: the threshold value

    Returns:
      edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
    """

    #
    # You code here
    #

    # D_x = 0.5 * np.array([1, 0, -1])
    # D_x = np.reshape(D_x, (1, 3))
    #
    # D_y = 0.5 * np.array([1, 0, -1])
    # D_y = np.reshape(D_y, (3, 1))
    #
    #
    # D2_x=ndimage.convolve(Ix,D_x)
    # D2_y=ndimage.convolve(Iy,D_y)
    # GM = np.sqrt(D2_x + D2_y)
    #######Gaussian filter
    # y_Gaussian = 0.5 * np.array([0.5395, 1, 0.5395])
    # y_Gaussian = np.reshape(y_Gaussian, (3, 1))
    # D_x = 0.5 * np.array([1, 0, -1])
    # D_x = np.reshape(D_x, (1, 3))
    # DF_x = y_Gaussian * D_x
    #
    # x_Gaussian = 0.5 * np.array([0.5395, 1, 0.5395])
    # x_Gaussian = np.reshape(x_Gaussian, (1, 3))
    # D_y = 0.5 * np.array([1, 0, -1])
    # D_y = np.reshape(D_y, (3, 1))
    # DF_y = x_Gaussian * D_y
    # D2_x=ndimage.convolve(Ix,DF_x)
    # D2_y=ndimage.convolve(Iy,DF_y)

    ## 3
    GM = np.sqrt(Ix ** 2 + Iy ** 2)
    # m_GM=np.ma.masked_array(GM,mask=(GM<thr))
    with np.nditer(GM, op_flags=['readwrite']) as it:
        for x in it:
            if (x < thr):
                x[...] = 0

    return GM


def nonmaxsupp(edges, Ix, Iy):
    """ Performs non-maximum suppression on an edge map.

    Args:
      edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
      Ix, Iy: filtered images

    Returns:
      edges2: edge map where non-maximum edges are suppressed
    """

    # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]

    # You code here
    size = np.shape(edges)
    edges2 = edges

    def top_to_bottom(index):
        if (index[0] == 0):
            if (edges[index] < edges[index[0]+1, index[1]]):
                return True
        if (0 < index[0] < size[0] - 1):
            if ((edges[index] < edges[index[0] - 1, index[1]]) | (edges[index] < edges[index[0] + 1, index[1]])):
                    return True
        if (index[0] == size[0] - 1):
            if (edges[index] < edges[index[0] - 1, index[1]]):
                 return True
        return False

    # handle left-to-right edges: theta in (-22.5, 22.5]

    # You code here
    def left_to_right(index):
        if (index[1] == 0):
            if (edges[index] < edges[index[0], index[1] + 1]):
                return True
        if 0 < index[1] < size[-1] - 1:
            if  ((edges[index] < edges[index[0], index[1] - 1]) | (edges[index] < edges[index[0], index[1] + 1])):
                return True
        if (index[1] == size[-1] - 1):
            if (edges[index] < edges[index[0], index[1] - 1]):
                return True
        return False

    # handle bottomleft-to-topright edges: theta in (22.5, 67.5]

    # Your code here
    def bottomleft_to_topright(index, ix, iy):
        offset = int(np.fix(iy / ix))
        # offset = offset[0]
        pixel = edges[index]
        interplot0=0 # keep edge candidate on boarder of image, infinite remove
        interplot1=0
        # topright
        if(((index[1]+1) < size[1]) & ((index[0] - offset - 1)>=0)):
            pixel_nextColume_offset = edges[index[0] - offset, index[1] + 1]
            pixel_nextColume_offset_above = edges[index[0] - offset - 1, index[1] + 1]
            interplot0 = pixel_nextColume_offset + (iy / ix - offset) * (
                        pixel_nextColume_offset_above - pixel_nextColume_offset)
        # bottomleft
        if(((index[1] - 1)>=0) & ((index[0] + offset + 1) < size[0])):
            pixel_lastColume_offset = edges[index[0] + offset, index[1] - 1]
            pixel_lastColume_offset_down = edges[index[0] + offset + 1, index[1] - 1]
            interplot1 = pixel_lastColume_offset + (iy / ix-offset) * (
                        pixel_lastColume_offset_down - pixel_lastColume_offset)

        if (pixel < interplot0) | (pixel < interplot1):
            return True
        return False

    # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]

    # Your code here
    def topleft_to_bottomright(index, ix, iy):
        offset = int(np.fix(iy / ix))
        offset = np.absolute(offset)
        pixel = edges[index]

        interplot0 = 0  # keep edge candidate on boarder of image, infinite remove
        interplot1 = 0

        # bottomright
        if(((index[0] + offset + 1) < size[0]) & ((index[1] + 1) < size[1])):
            pixel_nextColume_offset = edges[index[0] + offset, index[1] + 1]
            pixel_nextColume_offset_down = edges[index[0] + offset + 1, index[1] + 1]
            interplot0 = pixel_nextColume_offset + (-iy / ix - offset) * (
                    pixel_nextColume_offset_down - pixel_nextColume_offset)
        # topleft
        if (((index[0] - offset - 1) >= 0) & ((index[1] - 1) >=0)):
            pixel_lastColume_offset = edges[index[0] - offset, index[1] - 1]
            pixel_lastColume_offset_above = edges[index[0] - offset - 1, index[1] - 1]
            interplot1 = pixel_lastColume_offset + ( - iy / ix-offset) * (
                        pixel_lastColume_offset_above - pixel_lastColume_offset)

        if (pixel < interplot0 )|( pixel < interplot1):
            return True
        return False
    # i=0;
    with np.nditer(edges2, flags=['multi_index'], op_flags=['readwrite']) as it:
        for x in it:
            if x == 0:
                continue
            # cannot reach here 

            # i=i+1
            # print(i)
            index = it.multi_index
            iy = Iy[index]
            ix = Ix[index]
            # change from degree to radian
            angle = np.arctan2(iy, ix)* 180 / np.pi
            bool=(np.absolute(angle) > 67.5)

            if (np.absolute(angle) > 67.5):
                if top_to_bottom(index):
                    x[...] = 0
            if ((22.5 <= angle <= 67.5)):
                if bottomleft_to_topright(index, ix, iy):
                    x[...] = 0
            if (-22.5 <= angle <= 22.5):
                if left_to_right(index):
                    x[...] = 0
            if (-67.5 <= angle <= -22.5):
                if topleft_to_bottomright(index, ix, iy):
                    x[...] = 0

    return edges2
