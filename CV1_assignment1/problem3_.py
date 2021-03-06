import numpy as np
import matplotlib.pyplot as plt


# Plot 2D points
def displaypoints2d(points):
    plt.figure(0)
    plt.plot(points[0, :], points[1, :], '.b')
    plt.xlabel('Screen X')
    plt.ylabel('Screen Y')


# Plot 3D points
def displaypoints3d(points):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0, :], points[1, :], points[2, :], 'b')
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")


def cart2hom(points):
    """ Transforms from cartesian to homogeneous coordinates.

    Args:
      points: a np array of points in cartesian coordinates

    Returns:
      points_hom: a np array of points in homogeneous coordinates
    """

    #
    # You code here
    #
    ones = np.ones((1, np.shape(points)[1]))
    hom = np.vstack((points, ones))
    return hom


def hom2cart(points):
    """ Transforms from homogeneous to cartesian coordinates.

    Args:
      points: a np array of points in homogenous coordinates

    Returns:
      points_hom: a np array of points in cartesian coordinates
    """

    #
    # You code here
    #
    cart = points[0:-1, :] / points[-1, :]
    return cart


def gettranslation(v):
    """ Returns translation matrix T in homogeneous coordinates for translation by v.

    Args:
      v: 3d translation vector

    Returns:
      T: translation matrix in homogeneous coordinates
    """

    #
    # You code here
    #
    temp = np.reshape(v, (3, 1))
    hom = np.vstack((temp, 0))
    return hom

def getxrotation(d):
    """ Returns rotation matrix Rx in homogeneous coordinates for a rotation of d degrees around the x axis.

    Args:
      d: degrees of the rotation

    Returns:
      Rx: rotation matrix
    """

    #
    # You code here
    #
    return np.array([[1, 0, 0],
                     [0, np.cos(d), -np.sin(d)],
                     [0, np.sin(d), np.cos(d)]])


def getyrotation(d):
    """ Returns rotation matrix Ry in homogeneous coordinates for a rotation of d degrees around the y axis.

    Args:
      d: degrees of the rotation

    Returns:
      Ry: rotation matrix
    """

    #
    # You code here
    #
    return np.array([[np.cos(d), 0, np.sin(d)],
                     [0, 1, 0],
                     [-np.sin(d), 0, np.cos(d)]])


def getzrotation(d):
    """ Returns rotation matrix Rz in homogeneous coordinates for a rotation of d degrees around the z axis.

    Args:
      d: degrees of the rotation

    Returns:
      Rz: rotation matrix
    """

    #
    # You code here
    #
    return np.array([[np.cos(d), -np.sin(d), 0],
                     [np.sin(d), np.cos(d), 0],
                     [0, 0, 1]])


def getcentralprojection(principal, focal):
    """ Returns the (3 x 4) matrix L that projects homogeneous camera coordinates on homogeneous
    image coordinates depending on the principal point and focal length.

    Args:
      principal: the principal point, 2d vector
      focal: focal length

    Returns:
      L: central projection matrix
    """

    #
    # You code here
    #
    return np.array([[focal, 0, principal[0], 0],
                     [0, focal, principal[1], 0],
                     [0, 0, 1, 0]])


def getfullprojection(T, Rx, Ry, Rz, L):
    """ Returns full projection matrix P and full extrinsic transformation matrix M.

    Args:
      T: translation matrix
      Rx: rotation matrix for rotation around the x-axis
      Ry: rotation matrix for rotation around the y-axis
      Rz: rotation matrix for rotation around the z-axis
      L: central projection matrix

    Returns:
      P: projection matrix
      M: matrix that summarizes extrinsic transformations
    """

    #
    # You code here
    #
    R = Rz @ (Rx @ Ry)
    t = -R @ T[0:-1]
    temp = np.hstack((R, t))
    M = np.vstack((temp, [0, 0, 0, 1]))
    P = L @ M
    return P, M


def projectpoints(P, X):
    """ Apply full projection matrix P to 3D points X in cartesian coordinates.

    Args:
      P: projection matrix
      X: 3d points in cartesian coordinates

    Returns:
      x: 2d points in cartesian coordinates
    """

    #
    # You code here
    #
    temp_x = cart2hom(X)
    temp = P @ temp_x
    return hom2cart(temp)


def loadpoints():
    """ Load 2D points from obj2d.npy.

    Returns:
      x: np array of points loaded from obj2d.npy
    """

    #
    # You code here
    #
    points = np.load('data/obj2d.npy')

    return points


def loadz():
    """ Load z-coordinates from zs.npy.

    Returns:
      z: np array containing the z-coordinates
    """

    #
    # You code here
    #
    return np.load('data/zs.npy')


def invertprojection(L, P2d, z):
    """
    Invert just the projection L of cartesian image coordinates P2d with z-coordinates z.

    Args:
      L: central projection matrix
      P2d: 2d image coordinates of the projected points
      z: z-components of the homogeneous image coordinates

    Returns:
      P3d: 3d cartesian camera coordinates of the points
    """

    #
    # You code here
    #
    size_points = np.shape(P2d)
    ones = np.ones((1, size_points[1]))
    hom_points = np.vstack((P2d, ones))
    # check
    z_points = hom_points * z  # check
    # p_xy=L[0:2,2] #slicing construct no array
    # ??? why get [8, -10]
    p_xy = L[np.arange(2), 2]  # Integer array indexing

    p_xy_reshape = np.reshape(np.append(p_xy, 0), (3, 1))
    z_p_xy = p_xy_reshape * z
    inv_principal = z_p_xy + z_points
    L_truncted = L[np.arange(3), 0:3]
    P3d = np.linalg.inv(L_truncted) @ inv_principal

    return P3d


def inverttransformation(M, P3d):
    """ Invert just the model transformation in homogeneous coordinates
    for the 3D points P3d in cartesian coordinates.

    Args:
      M: matrix summarizing the extrinsic transformations
      P3d: 3d points in cartesian coordinates

    Returns:
      X: 3d points after the extrinsic transformations have been reverted
    """

    #
    # You code here
    #
    ones = np.ones((1, np.shape(P3d)[1]))
    p4d_C = np.vstack((P3d, ones))
    p4d_W = np.linalg.inv(M) @ p4d_C

    # p3d_W=p4d_W[0:-1,:]
    return p4d_W


def p3multiplecoice():
    '''
    Change the order of the transformations (translation and rotation).
    Check if they are commutative. Make a comment in your code.
    Return 0, 1 or 2:
    0: The transformations do not commute.
    1: Only rotations commute with each other.
    2: All transformations commute.
    '''

    return 0
