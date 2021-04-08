from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter

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

    # takes the path to an image file
    image = Image.open("./data/a2p1.png")
    image_arr = np.array(image)
    _range = np.max(image_arr) - np.min(image_arr)
    # returns the image as an array of floating point values between 0 and 1
    return (image_arr - np.min(image_arr)) / _range
    


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

    m, n = fsize[0], fsize[1]
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

    m, n = fsize[0], fsize[1]
    # Just make matrixes
    x = np.arange(-m/2+0.5,m/2)
    y = np.arange(-n/2+0.5,n/2)
    # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    # Expand rank
    X,Y = np.meshgrid(x, y, sparse = True)
    
    s = np.zeros((fsize[0],fsize[1]))
    for i in range(0,fsize[0]):
        for j in range(0,fsize[1]):
            if i>= j :
                s[i][j] = binom(i,j)
    
    return s/np.sum(s)

   



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
    img_result = convolve(img,f)

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

    a = np.zeros((img.shape[0], 1))
    b = np.zeros((1, img.shape[1]))
    c = np.row_stack((img, b))
    d = np.column_stack((img, a))
    img_result = convolve(img, f)
    return img_result
    

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

    image_list = []

    # downsample image
    image = downsample2(img,f)
    image_list.append(image)
    for i in range(0, nlevel-1):
        image = downsample2(image,f)
        image_list.append(image)


    # image1 = downsample2(img,f)
    # image2 = downsample2(image1,f)
    # image3 = downsample2(image2,f)

    # image_list = [image1, image2, image3]

    return image_list







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

    # recover image
    image1 = upsample2(gpyramid[0],f)
    image2 = upsample2(image1,f)
    image3 = upsample2(image2,f)
    image4 = upsample2(image3,f)
    image5 = upsample2(image4,f)
    image6 = upsample2(image5,f)

    image_list = [image1, image2, image3, image4, image5, image6]

    return image_list



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

    blurred_f = gaussian_filter(lpyramid[0], 3)
    filter_blurred_f = gaussian_filter(blurred_f, 1)
    alpha = 30
    
    blurred_f = lpyramid[0] - alpha * (blurred_f - filter_blurred_f)

    l = np.pad(blurred_f,((128,128),(128,128)),'constant', constant_values=(0))
    image = convolve(l,f)
    # clipped to [0, 1]
    image = image*alpha/np.sum(image)
    
    return image



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
    
    sharpened_list = []

    for i in range(0, len(lpyramid)):
        blurred_f = gaussian_filter(lpyramid[i], 3)
        filter_blurred_f = gaussian_filter(blurred_f, 1)
        alpha = 30
        sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
        sharpened_list.append(sharpened)

    return sharpened_list


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

    # numpy.pad
    if(pyramid[1].shape!=(256,256) and pyramid[1].shape!=(512,512)):
        pyramid1 = np.pad(pyramid[1],((64,64),(64,64)),'constant', constant_values=(0))
        pyramid2 = np.pad(pyramid[2],((96,96),(96,96)),'constant', constant_values=(0))
        pyramid3 = np.pad(pyramid[3],((112,112),(112,112)),'constant', constant_values=(0))
        pyramid4 = np.pad(pyramid[4],((120,120),(120,120)),'constant', constant_values=(0))
        pyramid5 = np.pad(pyramid[5],((124,124),(124,124)),'constant', constant_values=(0))
    
        concated_img=np.concatenate((pyramid[0],pyramid1),axis=1) 
        concated_img2=np.concatenate((concated_img,pyramid2),axis=1)  
        concated_img3=np.concatenate((concated_img2,pyramid3),axis=1)  
        concated_img4=np.concatenate((concated_img3,pyramid4),axis=1)  
        concated_img5=np.concatenate((concated_img4,pyramid5),axis=1)  
        return concated_img5

    elif len(pyramid) == 3:
        pyramid1 = pyramid[1]
        pyramid2 = pyramid[2]
        concated_img=np.concatenate((pyramid[0],pyramid1),axis=1) 
        concated_img2=np.concatenate((concated_img,pyramid2),axis=1)  
        return concated_img2
        
    else:
        pyramid1 = pyramid[1]
        pyramid2 = pyramid[2]
        pyramid3 = pyramid[3]
        pyramid4 = pyramid[4]
        pyramid5 = pyramid[5]

        concated_img=np.concatenate((pyramid[0],pyramid1),axis=1) 
        concated_img2=np.concatenate((concated_img,pyramid2),axis=1)  
        concated_img3=np.concatenate((concated_img2,pyramid3),axis=1)  
        concated_img4=np.concatenate((concated_img3,pyramid4),axis=1)  
        concated_img5=np.concatenate((concated_img4,pyramid5),axis=1)  
    
        return concated_img5
