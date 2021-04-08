import numpy as np
import os
from PIL import Image


def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images.
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        imgs: (N, H, W) numpy array
    """
    
    #
    # You code here
    #

    # first, search for files with extension
    images = []
    for root, dirs, files in os.walk(path):
        images += [os.path.join(root, fn) for fn in files if fn.endswith(ext)]
    
    print("Found {} images".format(len(images)))

    # next, load the images into an array
    all_images = []
    if len(images) > 0:
        for image_fn in images:
            img_arr = np.array(Image.open(image_fn))
            all_images.append(img_arr)
    
    imgs = np.stack(all_images, 0).astype(np.float)

    return imgs



def vectorize_images(imgs):
    """Turns an  array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images into
    an (N, M) array where M=H*W is the image dimension.
    
    Args:
        imgs: (N, H, W) numpy array
    
    Returns:
        x: (N, M) numpy array
    """
    
    #
    # You code here
    #
    
    # Vorlesung l4-pca Seite 10
    [groups ,rows, cols] = imgs.shape
    # print(groups,rows, cols)

    x = imgs.reshape(len(imgs), -1)
    return x

def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an numpy array with N M-dimensional features
    
    Returns:
        mean_face: (M,) numpy array representing the mean face
        u: (M, M) numpy array, bases with D principal components
        cumul_var: (N, ) numpy array, corresponding cumulative variance
    """

    #
    # You code here
    #

    # center images
    mean_face = X.mean(0)
    X = X - mean_face
    # Vorlesung l4-pca Seite 42
    # The left-singular vectors give us the eigenvectors of the covariance matrix.
    u, s, v = np.linalg.svd(X.T)

    N = X.shape[0]
    cumul_var = s**2 / N
    for i in range(cumul_var.shape[0] - 1):
        # cumulative variance
        cumul_var[i+1] = cumul_var[i] + cumul_var[i+1]
    return mean_face, u, cumul_var



def basis(u, cumul_var, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) numpy array containing principal components.
        For example, i'th vector is u[:, i]
        cumul_var: (N, ) numpy array, variance along the principal components.
    
    Returns:
        v: (M, D) numpy array, contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """
    
    #
    # You code here
    #
    d = 0
    var_cm = 0
    # make sure d is at least 1
    p = max(p, 1e-8)
    while var_cm / cumul_var[-1] < p and d < len(cumul_var):
        var_cm = cumul_var[d]
        d += 1
    
    v = u[:, :d]
    return v


def compute_coefficients(face_image, mean_face, u):
    """Computes the coefficients of the face image with respect to
    the principal components u after projection.
    
    Args:
        face_image: (M, ) numpy array (M=h*w) of the face image a vector
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        a: (D, ) numpy array, containing the coefficients
    """
    
    #
    # You code here
    #

    a = np.matmul(face_image - mean_face, u)
    return a


def reconstruct_image(a, mean_face, u):
    """Reconstructs the face image with respect to
    the first D principal components u.
    
    Args:
        a: (D, ) numpy array containings the image coefficients w.r.t
        the principal components u
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (M, ) numpy array, projected vector of face_image on 
        principal components
    """
    
    #
    # You code here
    #
    image_out = np.matmul(u, a) + mean_face
    return image_out


def compute_similarity(Y, x, u, mean_face):
    """Compute the similarity of an image x to the images in Y
    based on the cosine similarity.

    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) image we would like to retrieve
        u: (M, D) bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector

    Returns:
        sim: (N, ) numpy array containing the cosine similarity values
    """

    #
    # You code here
    #
    
    target = np.matmul(x - mean_face, u)
    target = target / np.linalg.norm(target)
    db = np.matmul(Y - mean_face, u)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    sim = np.matmul(db, target)

    return sim

def search(Y, x, u, mean_face, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) numpy array, image we would like to retrieve
        u: (M, D) numpy arrray, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M) numpy array containing the top_n most similar images
        sorted by similarity
    """

    #
    # You code here
    #

    sim = compute_similarity(Y, x, u, mean_face)
    top = np.argsort(-sim)[:top_n]
    Y_out = Y[top]
    return Y_out


def interpolate(x1, x2, u, mean_face, n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (M, ) numpy array, the first image
        x2: (M, ) numpy array, the second image
        u: (M, D) numpy array, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        n: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate n equally-spaced points on a line
    
    Returns:
        Y: (n, M) numpy arrray, interpolated results.
        The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """

    #
    # You code here
    #
    
    a1 = compute_coefficients(x1, mean_face, u)
    a2 = compute_coefficients(x2, mean_face, u)

    ims = [reconstruct_image(t * a1 + (1 - t) * a2, mean_face, u) for t in np.linspace(0, 1, n)]
    Y = np.stack(ims, 0)
    return Y