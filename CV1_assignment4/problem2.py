from functools import partial
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.ndimage import convolve


####################
# Provided functions
####################


conv2d = partial(convolve, mode="mirror")


def gauss2d(fsize, sigma):
    """ Create a 2D Gaussian filter

    Args:
        fsize: (w, h) dimensions of the filter
        sigma: width of the Gaussian filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def downsample(img, fsize=(5, 5), sigma=1.4):
    """
    Downsampling an image by a factor of 2

    Args:
        img: image as (h, w) np.array
        fsize and sigma: parameters for Gaussian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as (h/2, w/2) np.array
    """
    g_k = gauss2d(fsize, sigma)
    img = conv2d(img, g_k)
    return img[::2, ::2]


def gaussian_pyramid(img, nlevels=3, fsize=(5, 5), sigma=1.4):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        fsize: gaussian kernel size
        sigma: sigma of gaussian kernel

    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    pyramid = [img]
    for i in range(0, nlevels - 1):
        pyramid.append(downsample(pyramid[i], fsize, sigma))

    return pyramid


def resize(arr, shape):
    """ Resize an image to target shape

    Args:
        arr: image as (h, w) np.array
        shape: target size (h', w') as tuple

    Returns:
        resized image as (h', w') np.array
    """
    return np.array(Image.fromarray(arr).resize(shape[::-1]))


######################
# Basic Lucas-Kanade #
######################


def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives

    Args:
        im1: first image as (h, w) np.array
        im2: second image as (h, w) np.array

    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
                    as (h, w) np.array
    """

    #
    # You code here
    #
    def derivative_filters():
        """ Create derivative filters for x and y direction

        Returns:
            fx: derivative filter in x direction
            fy: derivative filter in y direction
        """
        fx =  np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]])
        fy = fx.transpose()
        return fx, fy
    
    mode = "mirror"

    fx, fy = derivative_filters()

    # first derivatives
    I_x = conv2d(im1, fx, cval=0)
    I_y = conv2d(im1, fy, cval=0)
    I_t = im1 - im2

    return I_x, I_y, I_t



def compute_motion(Ix, Iy, It, patch_size=15):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t each as (h, w) np.array
        patch_size: specifies the side of the square region R in Eq. (1)
    Returns:
        u: optical flow in x direction as (h, w) np.array
        v: optical flow in y direction as (h, w) np.array
    """

    #
    # You code here
    #

    # solve linear equation
    # ImportError: DLL load failed while importing _arpack: 找不到指定的程序。
    # from scipy.optimize import fsolve
    # def func(i):
    #     u, v = i[0], i[1]
    #     return (
    #     (u * Ix + v * Iy + It) * Ix,
    #     (u * Ix + v * Iy + It) * Iy
    #     )
    
    # r = fsolve(func, [0, 0])

    # return r[0], r[1]

    # Pre-calculation procedure of the matrix which we need to get u and v in calculate_uv
    def pre_calc(Ix, Iy, It):

        Xss = np.multiply(Ix, Ix)
        Yss = np.multiply(Iy, Iy)
        XYs = np.multiply(Ix, Iy)
        XTs = np.multiply(Ix, It)
        YTs = np.multiply(Iy, It)

        return Xss, Yss, XYs, XTs, YTs
    
    # Calculates u and v for each pixels.
    def calculate_uv(x,y,patch_size, Xss, Yss, XYs, XTs, YTs, m, n):

        t = patch_size//2

        X_ssum, Y_ssum, XY_sum, XT_sum, YT_sum = 0, 0, 0, 0, 0

        #For each pixels, calculate sum of each 15*15 window
        for i1 in range(-t, t+1):
            for j1 in range(-t, t+1):
                xc = x+i1
                yc = y+j1
                #i, j = padding(x, y, xc,yc,m,n)
                X_ssum += Xss[xc][yc]
                Y_ssum += Yss[xc][yc]
                XY_sum += XYs[xc][yc]
                XT_sum += XTs[xc][yc]
                YT_sum += YTs[xc][yc]

        A = np.array([[X_ssum, XY_sum], [XY_sum, Y_ssum]])
        B = np.array([-XT_sum, -YT_sum])

        #Compute result by A^(-1)*B
        result = np.linalg.inv(A)@B.T

        return result[0], result[1]


    u = np.empty_like(Ix)
    v = np.empty_like(Iy)

    t = patch_size//2

    Ix_pad = np.pad(Ix, t, mode='reflect')
    Iy_pad = np.pad(Iy, t, mode='reflect')
    It_pad = np.pad(It, t, mode='reflect')

    m, n = Ix.shape
    Xss, Yss, XYs, XTs, YTs = pre_calc(Ix_pad, Iy_pad, It_pad)

    for i in range (t, m+t):
        for j in range (t, n+t):
            uv, vv = calculate_uv(i,j,patch_size,Xss, Yss, XYs, XTs, YTs, m, n)
            u[i-t][j-t] = uv
            v[i-t][j-t] = vv

    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        im: input image as (h, w) np.array
        u, v: optical flow in x and y direction each as (h, w) np.array

    Returns:
        im_warp: warped image as (h, w) np.array
    """

    #
    # You code here
    #

    im_warp = np.empty_like(im)
    #
    # Your code here
    #

    m, n = im.shape
    data1 = []
    data2 = []
    for i in range(m):
        for j in range(n-1):
            data1.append((i+u[i][j], j+v[i][j]))
            data2.append(im[i][j])

    grid_x, grid_y = np.mgrid[0:m, 0:n]
    im_warp = interpolate.griddata(np.asarray(data1), np.asarray(data2), (grid_x, grid_y), method = 'linear', fill_value = 0)

    return im_warp


def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade.
    Args:
        im1, im2: Images as (h, w) np.array

    Returns:
        Cost as float scalar
    """

    #
    # You code here
    #

    # l9-motion Seite 52
    Ix, Iy, It = compute_derivatives(im1, im2)
    u, v = compute_motion(Ix, Iy, It)

    cost = np.sum(np.square(np.multiply(u, Ix) + np.multiply(v, Iy) + It))

    return cost




###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(pyramid1, pyramid2, n_iter=10):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.

    Args:
        pyramid1, pyramid2: Gaussian pyramids corresponding to
                            im1 and im2, in fine to coarse order
        n_iter: number of refinement iterations

    Returns:
        u: OF in x direction as np.array
        v: OF in y direction as np.array
    """

    #
    # You code here
    #

    u = np.zeros_like(pyramid1)
    v = np.zeros_like(pyramid2)

    pyramid_len = len(pyramid1)

    for i in range(n_iter):
        for k in range(pyramid_len):
            img1 = pyramid1[k]
            img2 = pyramid2[k]
            if i !=0 or k !=0:
                img1 = warp(img1, u, v)
            Ix, Iy, It = compute_derivatives(img1,img2)
            # get optical flow in x and y direction
            u, v = compute_motion(Ix, Iy, It)
            if k !=pyramid_len-1:
                # https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
                u = u.repeat(2, axis=0).repeat(2, axis=1)
                v = v.repeat(2, axis=0).repeat(2, axis=1)

            if k == pyramid_len-1 and i !=n_iter-1:
                u = downsample(u)
                v = downsample(v)
                
    return u, v