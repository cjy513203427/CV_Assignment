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

  g_fx = 0.5 * np.array([1, 0, -1])
  g_fx = np.reshape(g_fx,(1,3))

  g_fy = 0.5 * np.array([1, 0, -1])
  g_fy = np.reshape(g_fy,(3,1))


def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

  #
  # You code here
  #
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

  # handle left-to-right edges: theta in (-22.5, 22.5]

  # You code here

  # handle bottomleft-to-topright edges: theta in (22.5, 67.5]

  # Your code here

  # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]

  # Your code here
