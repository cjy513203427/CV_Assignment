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
    im = Image.open(path)
    a = np.asarray(im)
    a_normal = a / np.linalg.norm(a)
    return a_normal


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
    filter_Gaussian = []

    row_index = (fsize[0] if fsize[0] % 2 == 0 else fsize[0] - 1) * np.linspace(-0.5, 0.5, fsize[0])
    col_index = (fsize[1] if fsize[1] % 2 == 0 else fsize[1] - 1) * np.linspace(-0.5, 0.5, fsize[1])

    for row in row_index:
        for col in col_index:
            filter_Gaussian.append((1 / (2 * np.pi * np.power(sigma, 2))) * \
                                   np.exp(-(np.power(row, 2) + np.power(col, 2)) / (2 * np.power(sigma, 2))))
    filter_Gaussian = np.reshape(filter_Gaussian, fsize)
    # max=np.amax(filter_Gaussian)
    filter_Gaussian_normal = filter_Gaussian * (1 / np.sum(filter_Gaussian))
    return filter_Gaussian_normal


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
    row_index = range(0, fsize[0] - 1)
    col_index = range(0, fsize[1] - 1)
    weights_row = []
    weights_col = []
    weights = np.empty(fsize)

    for row in row_index:
        weight = binom(fsize[0] - 1, row)
        weights_row.append(weight)
    for col in col_index:
        weight = binom(fsize[1] - 1, col)
        weights_col.append(weight)
    sum_weights_row = np.sum(weights_row)
    sum_weights_col = np.sum(weights_col)

    for row in row_index:
        for col in col_index:
            weights[row][col] = weights_row[row] * weights_col[col] / (sum_weights_row * sum_weights_col)
    return weights * (1 / np.sum(weights))


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
    img_filtered = convolve(img, f, mode='constant')
    downsample_row_img = []
    for i in range(0, np.shape(img)[0]):
        if i % 2 == 0:
            downsample_row_img.append(img_filtered[i])

    downsample_img_array = np.asarray(downsample_row_img)
    delete_index = []
    for i in range(0, np.shape(img)[1]):
        if i % 2 == 0: delete_index.append(i)
    downsample_img_array = np.delete(downsample_img_array, delete_index, 1)

    return downsample_img_array


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
    shape = np.shape(img)
    img_upsample = deepcopy(img)
    r = range(0, shape[0])
    img_upsample = np.insert(img_upsample, r, 0, axis=0)
    c = range(0, shape[1])
    img_upsample = np.insert(img_upsample, c, 0, axis=1)
    img_filtered = convolve(img_upsample, f, mode='constant')
    # showimage(img_filtered)
    return img_filtered


def showimage(img):
    plt.figure(dpi=150)
    plt.imshow(img, cmap="gray", interpolation="none")
    plt.axis("off")
    plt.show()


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
    pyramid = []
    for level in range(0, nlevel):
        if level == 0:
            pyramid.append(img)
        else:
            dowmsample = downsample2(pyramid[level - 1], f)
            pyramid.append(dowmsample)
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
    lpyramid = []
    gpyramid_flip = np.flip(gpyramid)
    for count, gpyramid_i in enumerate(gpyramid_flip):
        if count == 0:
            lpyramid.append(gpyramid_i)
        else:
            upsample = upsample2(gpyramid_flip[count - 1], f)
            lNew = gpyramid_i - upsample
            # lNew = upsample
            lpyramid.append(lNew)
    lpyramid_flip = np.flip(lpyramid)
    return lpyramid_flip.tolist()


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
    lpyramid_flip = np.flip(lpyramid)
    gpyramid = []
    for count, lpyramid_i in enumerate(lpyramid_flip):
        if count == 0:
            gpyramid.append(lpyramid_i)
            continue
        upsample = upsample2(gpyramid[count - 1], f)
        gpyramid_i = upsample + lpyramid_i
        gpyramid.append(gpyramid_i)
    return gpyramid[-1]


def amplifyhighfreq(lpyramid, l0_factor=1.5, l1_factor=1.5):
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
    lpyramid_temp = deepcopy(lpyramid)
    lpyramid_temp[0] *= l0_factor
    lpyramid_temp[1] *= l1_factor
    return lpyramid_temp


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
    height = np.shape(pyramid[0])[0]
    # width=0
    pyramid_temp = deepcopy(pyramid)
    composite = deepcopy(pyramid[0]) / np.amax(pyramid[0])
    # showimage(composite)
    for count, pyramid_i in enumerate(pyramid_temp):
        max = np.amax(pyramid_i)
        pyramid_i = pyramid_i / max
        # print(max)
        # showimage(pyramid_i)
        if count == 0:
            continue
        height_i = np.shape(pyramid_i)[1]
        width_i = np.shape(pyramid_i)[0]

        concatenate0 = np.zeros((height - height_i, width_i))
        pyramid_i_0 = np.vstack((pyramid_i, concatenate0))
        composite = np.hstack((composite, pyramid_i_0))

    return composite