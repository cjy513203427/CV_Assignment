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
    assert im1.shape == im2.shape

    dx = np.array([[0.5, 0, -0.5]])
    dy = dx.T

    Ix = conv2d(im1, dx)
    Iy = conv2d(im2, dy)
    # l9-motion Seite 32
    It = im1 - im2

    assert Ix.shape == im1.shape and Iy.shape == im1.shape and It.shape == im1.shape

    return Ix, Iy, It
    



def compute_motion(Ix, Iy, It, patch_size=15):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t each as (h, w) np.array
        patch_size: specifies the side of the square region R in Eq. (1)
    Returns:
        u: optical flow in x direction as (h, w) np.array
        v: optical flow in y direction as (h, w) np.array
    """
    assert Ix.shape == Iy.shape and Iy.shape == It.shape

    IxIy = Ix * Iy
    IxIx = Ix * Ix
    IyIy = Iy * Iy

    IxIt = Ix * It
    IyIt = Iy * It

    k = np.ones((patch_size, patch_size))

    Axy = conv2d(IxIy, k)
    Axx = conv2d(IxIx, k)
    Ayy = conv2d(IyIy, k)

    Bxt = -conv2d(IxIt, k)
    Byt = -conv2d(IyIt, k)

    # solving for the motion
    z = Axx * Ayy - Axy * Axy

    assert (np.abs(z) > 0).all()

    v = (Axx * Byt - Axy * Bxt) / z
    u = (Ayy * Bxt - Axy * Byt) / z

    assert u.shape == Ix.shape and v.shape == Ix.shape
    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        im: input image as (h, w) np.array
        u, v: optical flow in x and y direction each as (h, w) np.array

    Returns:
        im_warp: warped image as (h, w) np.array
    """

    assert im.shape == u.shape and u.shape == v.shape

    # lets prepare data for fitting
    h, w = im.shape

    x = np.arange(w, dtype=np.float)
    y = np.arange(h, dtype=np.float)

    xs, ys = np.meshgrid(x, y)

    xs_fwd = (xs + u).flatten()
    ys_fwd = (ys + v).flatten()

    points = np.stack([xs_fwd, ys_fwd], axis=-1)
    im_warp = interpolate.griddata(
        points, im.flatten(), (xs, ys), method="linear", fill_value=0
    )

    assert im_warp.shape == im.shape
    return im_warp


    


def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade.
    Args:
        im1, im2: Images as (h, w) np.array

    Returns:
        Cost as float scalar
    """

    assert im1.shape == im2.shape

    h, w = im1.shape
    d = im1 - im2
    d = (d * d).sum() / (h * w)

    assert isinstance(d, float)
    return d




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

    im1 = pyramid1[0]
    nlevels = len(pyramid1)

    # initialize OF
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)

    # reverse gaussian pyramids
    pyramid1 = pyramid1[::-1]
    pyramid2 = pyramid2[::-1]

    for i in range(n_iter):
        # initialize increments for this iteration
        du = np.zeros(im1.shape)
        dv = np.zeros(im1.shape)

        for l, p1, p2 in zip(range(nlevels), pyramid1, pyramid2):

            # upscale the previous OF to current resolution
            du = 2 * resize(du, p1.shape)
            dv = 2 * resize(dv, p2.shape)

            # resize OF to current resolution
            iu = resize(u, p1.shape) / 2 ** (nlevels - l - 1)
            iv = resize(v, p1.shape) / 2 ** (nlevels - l - 1)

            # estimate of OF so far
            curr_u = iu + du
            curr_v = iv + dv

            p1_warped = warp(p1, curr_u, curr_v)
            cost_before = compute_cost(p1_warped, p2)

            Ix, Iy, It = compute_derivatives(p1_warped, p2)
            est_u, est_v = compute_motion(Ix, Iy, It)

            p1_warped = warp(p1, curr_u + est_u, curr_v + est_v)
            cost_after = compute_cost(p1_warped, p2)

            print("Cost: {:4.3e} -> {:4.3e}".format(cost_before, cost_after))

            du += est_u
            dv += est_v
        
        delta = max(np.abs(du).max(), np.abs(dv).max())

        u += du
        v += dv

        im1_warped = warp(im1, u, v)
        ssd = compute_cost(im1_warped, im2)
        print(f"Iteration {i + 1}: SSD = {ssd:4.3e}, delta = {delta:4.3e}")

    assert u.shape == im1.shape and v.shape == im1.shape
    return u,v

