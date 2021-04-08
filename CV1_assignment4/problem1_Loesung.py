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

    U, d, V = np.linalg.svd(A)
    d[-1] = 0
    A_hat = U @ np.diag(d) @ V
    return A_hat




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

    x1 = p1[:, 0:1]
    y1 = p1[:, 1:2]
    x2 = p2[:, 0:1]
    y2 = p2[:, 1:2]
    I = np.ones((x1.shape[0], 1))
    # Vorlesung l8-stereo Seite 21
    A = np.concatenate([x1*x2, y2*x1, x1, x2*y1, y1*y2, y1, x2, y2, I], axis=1)
    _, _, Vt = np.linalg.svd(A)
    F = Vt.T[:,-1].reshape((3,3))
    F = enforce_rank2(F)
    return F






def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """

    u1, T1 = condition_points(p1)
    u2, T2 = condition_points(p2)
    F = compute_fundamental(u1, u2)
    # Vorlesung l8-stereo Seite 22
    F = (T1.T).dot(F.dot(T2))
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

    n = p1.shape[0]
    P = np.concatenate([p1, np.ones((p1.shape[0], 1))], axis=1)
    l = P @ F.T
    # line coefficients: x^T * l = 0, with l = F*points = (a, b, c)^T
    # Vorlesung l8-stereo Seite 19
    a = l[:, 0]
    b = l[:, 1]
    c = l[:, 2]
    # first point of the line
    X1 = np.zeros(n)
    Y1 = (-c + a * X1) / b
    # last point of the line
    X2 = X1 + img.shape[1] - 1
    Y2 = -(c + a * X2) / b
    return X1, X2, Y1, Y2








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

    n = p1.shape[0]
    residuals = []
    for i in range(n):
        residuals.append(np.abs(p1[i,:] @ F @ p2[i,:]))
    max_residual = np.max(residuals)
    avg_residual = np.mean(residuals)
    return max_residual, avg_residual


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """
    
    # Vorlesung l8-stereo Seite 19
    V1, d, V2 = np.linalg.svd(F)
    e1 = V1[0:2, 2] / V1[2, 2]
    e2 = V2.T[0:2, 2] / V2[2, 2]
    return e1, e2