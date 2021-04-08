import numpy as np
import matplotlib.pyplot as plt


# Plot 2D points
def displaypoints2d(points):
  plt.figure(0)
  plt.plot(points[0,:],points[1,:], '.b')
  plt.xlabel('Screen X')
  plt.ylabel('Screen Y')


# Plot 3D points
def displaypoints3d(points):
  fig = plt.figure(1)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(points[0,:], points[1,:], points[2,:], 'b')
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

  # points = np.transpose(points)
  new_ele = [[1]]
  points_hom = np.append(points,new_ele, axis = 0)
  return points_hom



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
  # points = np.transpose(points)
  #hom2cart = points/points[-1]
  # Delete the last element
  points_hom = np.delete(points, -1, axis = 0)
  return points_hom




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

  # translation matrix:
  # [[1,0,vx],
  #  [0,1,vy],
  #  [0,0,1],
  T = np.array([[1,0,v[0]],[0,1,v[1]],[0,0,1]])
  return T


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
  Rx = np.array([[1,0,0],
                [0,np.cos(d),np.sin(d)],
                [0,-np.sin(d),np.cos(d)]]) 
  return Rx


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

  Ry = np.array([[np.cos(d),0,-np.sin(d)],
              [0,1,0],
              [np.sin(d),0,np.cos(d)]]) 
  return Ry



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
  Rz = np.array([[np.cos(d),np.sin(d),0],
              [-np.sin(d),np.cos(d),0],
              [0,0,1]]) 
  return Rz



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

  # [[f,0,px,0],
  #  [0,f,py,0],
  #  [0,0,1,0]] 
  L = np.zeros((3,4))
  L[0,0] = focal
  L[1,1] = focal
  L[0,2] = principal[0]
  L[1,2] = principal[1]
  L[2,2] = 1

  return L



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

  # P = K[R|T]
  # Make central projection matrix 3✖3
  L = np.delete(L, -1, axis=1)
  r1 = np.dot(Rx,Ry)
  r2 = np.dot(r1,Rz)
  # Merge rotation matrix and translation matrix
  T = T[:,[-1]]
  r3 = np.hstack((r2,T))
  P = np.dot(L,r3)
  # M = [R|T]
  s1 = np.dot(Rx,Ry)
  s2 = np.dot(s1,Rz)
  M = np.hstack((s2,T))
  return P,M

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

  # ignore projection matrix
  x = hom2cart(X)
  focal_length = 8
  x[0,0] = x[0,0]/focal_length
  x[1,0] = x[1,0]/focal_length

  return x





def loadpoints():
  """ Load 2D points from obj2d.npy.

  Returns:
    x: np array of points loaded from obj2d.npy
  """

  #
  # You code here
  #

  # Load arrays from ``.npy``
  x = np.load('data/obj2d.npy')
  return x


def loadz():
  """ Load z-coordinates from zs.npy.

  Returns:
    z: np array containing the z-coordinates
  """

  #
  # You code here
  #
  z = np.load('data/zs.npy')
  return z


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
  L = np.delete(L, 0, axis=1)
  L = np.delete(L, 0, axis=1)
  # P2d 2✖300
  r1 = np.dot(L,P2d)
  # Make z matrix 300✖1
  P3d = np.dot(r1,z.T)
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

  # Make P3d 4✖1
  P3d = cart2hom(P3d)
  r = np.dot(M,P3d)
  X = cart2hom(r)
  return X




def p3multiplecoice():
  '''
  Change the order of the transformations (translation and rotation).
  Check if they are commutative. Make a comment in your code.
  Return 0, 1 or 2:
  0: The transformations do not commute.
  1: Only rotations commute with each other.
  2: All transformations commute.
  '''

  return -1