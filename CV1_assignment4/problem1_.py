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
    T[0:2, 2] = -t
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
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    s_rank2 = s
    s_rank2[-1] = 0
    A_hat = u @ np.diag(s_rank2) @ vh
    return A_hat


def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordines in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """

    #
    # You code here
    #
    A_list = []
    for i, p in enumerate(p1):  # [0:8,:] pick 8 ponts
        temp = np.array(
            [p[0] * p2[i][0], p[1] * p2[i][0], p2[i][0], p[0] * p2[i][1], p[1] * p2[i][1], p2[i][1], p[0], p[1], 1])
        A_list.append(temp)
    A = np.asarray(A_list)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    V = vh.T
    F = np.reshape(V[:, -1], (3, 3))  # check!
    # F/=F[2,2]
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

    #
    # You code here
    #
    ps1, T1 = condition_points(p1)
    ps2, T2 = condition_points(p2)
    F_rank3 = compute_fundamental(ps1, ps2)
    F_hat = enforce_rank2(F_rank3)
    F = T2.T @ F_hat @ T1
    return F


def cart2hom(p_c):
    p_h = np.concatenate([p_c, np.ones((p_c.shape[0], 1))], axis=1)
    return p_h


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
    p1_homo = cart2hom(p1)
    normals = (F @ p1_homo.T).T  # normals: (n,3)
    # only check pixel coordinates on borders
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []  # 2 intersected points on borders
    img_H = img.shape[0]
    img_W = img.shape[1]

    # construct coordinates of pixels on borders
    p_coordinates_borders = []
    for row in range(0, img_H):
        for col in range(0, img_W):
            if (0 < row < img_H - 1 and 0 < col < img_W - 1):
                continue
            p_coordinates_borders.append(np.array([col, row, 1]))  # (x:width, y:height, 1)homogenous
    A_p_coordinates_borders = np.asarray(p_coordinates_borders)

    # dot product of pixel coordinates and normal vector
    dot_product_normals = []  # => (n, number of pixels)
    for normal in normals:  # n iterations
        dot_product_normal = []  # => (number of pixels, )
        for p in p_coordinates_borders:
            normal_vector = normal.flatten()
            dot_product_normal.append(np.dot(normal_vector, p))
        dot_product_normals.append(np.asarray(dot_product_normal))

    A_dot_product_normals = np.asarray(dot_product_normals)
    ind = np.argsort(np.abs(A_dot_product_normals),
                     axis=1)  # (n, number of pixels), sorts along second axis (--), dot_product can be negativ
    # min=np.min(np.abs(A_dot_product_normals), axis=1)
    # print(np.sort(np.abs(A_dot_product_normals[0])))
    # find coordinates of 2 pixels with smallest dot product
    for n, dot_product_ind in enumerate(
            ind):  # row, dot_product_ind[[0,1]] are the indices of the 2 2 pixels with smallest dot product
        ind1 = dot_product_ind[0]
        coor1 = p_coordinates_borders[ind1]
        X1.append(coor1[0])  # W
        Y1.append(coor1[1])  # H
        # print(A_dot_product_normals[n][ind1])

        ind2 = dot_product_ind[1]
        coor2 = p_coordinates_borders[ind2]
        X2.append(coor2[0])
        Y2.append(coor2[1])
        # print(A_dot_product_normals[n][ind2])

    A_X1 = np.asarray(X1)
    A_X2 = np.asarray(X2)
    A_Y1 = np.asarray(Y1)
    A_Y2 = np.asarray(Y2)
    return A_X1, A_X2, A_Y1, A_Y2


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
    residual = p1 @ F @ p2.T
    residual_abs = np.abs(residual)
    max_residual = np.max(residual_abs)
    avg_residual = np.mean(residual_abs)
    return max_residual, avg_residual


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
    
    u, s, vh1 = np.linalg.svd(F.T)
    e1 = vh1.T[:, -1]
    e1 /= e1[2]

    u, s, vh2 = np.linalg.svd(F)
    e2 = vh2.T[:, -1]
    e2 /= e2[2]
    return e1[0:2], e2[0:2]
