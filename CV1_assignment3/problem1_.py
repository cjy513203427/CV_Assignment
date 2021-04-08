import numpy as np
from scipy.ndimage import convolve, maximum_filter


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter
    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction
    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix
    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """

    #
    # You code here
    #
    img_filtered = convolve(img, gauss,mode='mirror')
    I_x=convolve(img_filtered, fx,mode='mirror')
    I_y= convolve(img_filtered, fy, mode='mirror')
    I_xx = convolve(I_x, fx, mode='mirror')
    I_yy = convolve(I_y, fy, mode='mirror')
    I_xy=convolve(I_x, fy, mode='mirror')
    return I_xx,I_yy,I_xy


def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function
    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor
    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """

    #
    # You code here
    #
    # criterion=np.power(sigma,4)*(I_xx@I_yy-I_xy@I_xy)
    #
    # H1=np.hstack((I_xx,I_xy))
    # H2=np.hstack((I_xy,I_yy))
    # H=np.vstack((H1,H2))
    # det_H=np.linalg.det(H)
    # criterion = np.power(sigma, 4) * det_H

    criterion = np.power(sigma, 4) * (np.multiply(I_xx,I_yy) - np.multiply(I_xy, I_xy))
    return criterion

def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points
        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """

    #
    # You code here
    #
    criterion_filtered = maximum_filter(criterion,size=5,mode='mirror')
    size = np.shape(criterion_filtered)
    with np.nditer(criterion_filtered, op_flags=['readwrite'], flags=['multi_index']) as it:
        for x in it:
            index=it.multi_index
            if (index[0]>(size[0]-1-5))or(index[0]<5)or(index[1]>(size[1]-1-5))or(index[1]<5):
                x[...]=0


    rows, cols = np.nonzero(criterion_filtered > threshold)

    return rows, cols