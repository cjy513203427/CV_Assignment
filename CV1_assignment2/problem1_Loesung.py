from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def loadimg(path):
    """ Load image file
    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """

    #
    # You code here
    #
    img = Image.open(path)
    # why not normalized?
    img = np.array(img)
    return img


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """

    #
    # You code here
    #

    m, n = fsize
    # Just make matrixes
    x = np.arange(-m/2+0.5,m/2)
    y = np.arange(-n/2+0.5,n/2)
    # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    # Expand rank
    X,Y = np.meshgrid(x, y, sparse = True)
    g = np.exp(-(X**2 + Y**2)/(2*sigma**2))
    # Normalization
    return g/np.sum(g)



def binomial2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """

    #
    # You code here
    #

    def binomialfactors(n):
        a = np.array([[binom(n-1,i) for i in range(0, n)]])
        return a
        
    x = binomialfactors(fsize[0])
    y = binomialfactors(fsize[1])
    f = y.T * x

    return f/np.sum(f)

   



def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """

    #
    # You code here
    #

    # filter with Gaussian filter
    img_result = convolve(img,f,mode = "mirror")

    # take every other row/column
    return img_result[::2,::2]


def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """

    #
    # You code here
    #

    up = np.zeros(2 * np.array(img.shape))
    up[::2,::2] = img
    # a scale factor is 4
    up = 4 * convolve(up, f, mode="mirror")

    return up
    

def gaussianpyramid(img, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    #
    # You code here
    #

    pyramid = [img]
    for i in range(0, nlevel-1):
        pyramid.append(downsample2(pyramid[i],f))
    
    return pyramid







def laplacianpyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    #
    # You code here
    #
    
    
    # reverse order of gaussian pyramid
    gpyramid = gpyramid[::-1]

    # build lapacian pyramid from coarse to fine
    lpyramid = [gpyramid[0]]
    for i in range(1, len(gpyramid)):
        # Vorlesung l3-edges_pyramids Seite 62
        lpyramid.append(gpyramid[i] - upsample2(gpyramid[i-1], f))

    # reverse lapalacian pyramid to correct order
    return lpyramid[::-1]

def reconstructimage(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """

    #
    # You code here
    #

    # start with coarsest level of Laplacian pyramid
    img = lpyramid[-1]

    # upsample and add next finer level
    for l in lpyramid[-2::-1]:
        img = l + upsample2(img, f)
    
    # make sure values of reconstructed image are inside bounds
    img = img.clip(0, 1)

    return img


def amplifyhighfreq(lpyramid, l0_factor=1.0, l1_factor=1.0):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """

    #
    # You code here
    #
    lpyramid = deepcopy(lpyramid)
    lpyramid[0] += l0_factor
    lpyramid[1] += l1_factor

    return lpyramid

def createcompositeimage(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
    """

    #
    # You code here
    #

    def normalize(img):
        return (img - img.min())/(img.max() - img.min())
    
    h = pyramid[0].shape[0]
    composite = [
        # array of rank 2 expands array
        np.pad(normalize(img), ((0, h - img.shape[0]), (0,0))) for img in pyramid
    ]
    composite = np.concatenate(composite, axis=1)

    return composite

    