import numpy as np
import matplotlib.pyplot as plt



def condition_points(points):
    """ Conditioning: Normalization of coordinates for numeric stability 
    by substracting the mean and dividing by half of the component-wise
    maximum absolute value.
    Args:
        points: (l, 3) numpy array containing unnormalized homogeneous coordinates.

    Returns:
        ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
        T: (3, 3) numpy array, transformation matrix for conditioning
    """
    t = np.mean(points, axis=0)[:-1]
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2,2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1)
    ps = points @ T.T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """

    #
    # You code here
    #

    # _, _, vh = np.linalg.svd(A, full_matrices = True)

    # setting the smallest eigenvalue to zero
    # A_hat = np.maximum(vh, 0)
    # vh[2, 2] = 0
    # return vh

    assert A.shape == (3, 3)
    
    #
    # Your code goes here
    #
    #First perform a SVD. 
    u,s,vt = np.linalg.svd(A)

    #Set the last singular value to 0.
    s[2] = 0

    #Re-construct matrix using modified s vector. 
    F_final = (u*s)@vt
    
    return F_final




def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """

    #
    # You code here
    #

    # Vorlesung l8-stereo Seite 18
    p = p1.T @ p2
    # get (3, 3) fundamental matrix
    u, F, vh  = np.linalg.svd(p, full_matrices = True, compute_uv = True)
    return u





def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """

    #
    # You code here
    #
    # Vorlesung l8-stereo Seite 21
    # It seems like "no use"
    F = compute_fundamental(p1, p2)
    f = F.flatten()
    # normalize
    f = f / np.sum(f)
    # Vorlesung l8-stereo Seite 23  Af = 0
    # A = [xx' yx' x' xy' yy' y' x y 1]
    # A = np.array([ p1[:, 0] * p2[:, 0], p1[:, 1] * p2[:, 0], p2[:, 0], p1[:, 0] * p2[:, 1],
    #               p1[:, 1] * p2[:, 1], p2[:, 1], p1[:, 0], p1[:, 1], 1])
    Ua, Da, Va = np.linalg.svd(F, full_matrices = True)
    # F~
    # F_ = np.reshape(Ua[:,8], (3,3))
    F_ = Ua
    Uf, Df, Vf = np.linalg.svd(F_, full_matrices = True)
    # make it rank 2
    # Df[2, 2] = 0
    Uf[2, 2] = 0
    # F-
    F__ = Uf @ F_ @ Vf
    # get transformation matrix for conditioning 
    _, T1 =  condition_points(p1)
    _, T2 = condition_points(p2)

    F = T2.T @ F__ @ T2

    return F

def draw_epipolars(F, p1, img):
    """ Computes the coordinates of the n epipolar lines (X1, Y1) on the left image border and (X2, Y2)
    on the right image border.
    Args:
        F: (3, 3) numpy array, fundamental matrix 
        p1: (n, 2) numpy array, cartesian coordinates of the point correspondences in the image
        img: (H, W, 3) numpy array, image data

    Returns:
        X1, X2, Y1, Y2: (n, ) numpy arrays containing the coordinates of the n epipolar lines
            at the image borders
    """

    #
    # You code here
    #

    # calculate line O1P1
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    # what is the function of img data
    x2 = p1[:, 1]
    y2 = p1[:, 0]

    k = (y2 - y1)/(x2 - x1)
    b = y1 - k*x1
    
    return x1,x2,y1,y2






def compute_residuals(p1, p2, F):
    """
    Computes the maximum and average absolute residual value of the epipolar constraint equation.
    Args:
        p1: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 1
        p2: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 2
        F:  (3, 3) numpy array, fundamental matrix

    Returns:
        max_residual: maximum absolute residual value
        avg_residual: average absolute residual value
    """

    #
    # You code here
    #

    g = 0
    m,n = p1.shape
    
    #Calculate the residual for each points. 
    for i in range(m):
        g += np.abs(p1[i] * F * p2[i].T)
    
    g = g/m

    return np.max(g), np.mean(g)


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    #
    # You code here
    #
    u, s, v = np.linalg.svd(F)

    e1 = u[0:2, 0]
    e2 = v[-2:, -1]

    return e1, e2