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
    imgs=[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            path=os.path.join(root, name)
            im = Image.open(path)
            a = np.asarray(im)
            imgs.append(a)
    imgs_array= np.asarray(imgs)
    return imgs_array


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
    vector=[]
    for img in imgs:
        vector.append(img.flatten())
    vector_array=np.asarray(vector)
    return vector_array

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
    N=np.shape(X)[0]
    mean_face=np.sum(X, axis=0)/N

    X_mean0=X-mean_face # x hat
    u, s, vh = np.linalg.svd(X_mean0.T, full_matrices=True) # must transpose to get basis dimension right
    variance = np.power(s,2)/N
    cumul_var=[]
    for count, var in enumerate(variance):
        if count==0:
            cumul_var.append(var)
        else:
            cumul_var.append(var+cumul_var[count-1])
    cumul_var_array=np.asarray(cumul_var)
    return mean_face, u, cumul_var_array









def basis(u, cumul_var, p=0.5):
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
    sum_var=cumul_var[-1]
    basis=[]
    for count, var in enumerate(cumul_var):
        if var<p*sum_var:
            basis.append(u[:,count])
        else:
            basis.append(u[:,count])
            break

    basis_array=np.asarray(basis).T
    return basis_array




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
    face_mean0=face_image-mean_face
    D=np.shape(u)[1]
    a=[]
    for i in range(0,D):
        u_i=u[:,i].flatten()
        dot_product_M=face_mean0*u_i
        dot_product=np.sum(dot_product_M)
        a.append(dot_product)
    a_array=np.asarray(a)
    return a_array






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
    image_out=np.zeros_like(mean_face)
    for count, a_i in enumerate(a):
        image_out += a_i*u[:,count]
    return image_out+mean_face


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
    x_mean0=x-mean_face
    Y_mean0=Y-mean_face
    x_descriptor=[]
    sim=[]

    for u_i in u.T:
        x_descriptor.append(np.sum(x_mean0*u_i))
    x_descriptor=np.asarray(x_descriptor)
    # The Frobenius norm
    x_descriptor_length = np.linalg.norm(x_descriptor)

    for y_i in Y_mean0:
        y_i_descriptor=[]
        for u_i in u.T:
            y_i_descriptor.append(np.sum(u_i*y_i))
        y_i_descriptor=np.asarray(y_i_descriptor)
        sim_i=np.inner(y_i_descriptor,x_descriptor)
        sim.append(sim_i/(x_descriptor_length * np.linalg.norm(y_i_descriptor)))

    sim_array=np.asarray(sim)
    return sim_array








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
    sim=compute_similarity(Y, x, u, mean_face)
    dtype=[('sim', float), ('image', np.ndarray)] #? np.ndarray need predefined size? <= no need
    values=[]

    for count, sim_i in enumerate(sim):
        values.append((sim_i,Y[count]))


    toSort=np.array(values, dtype=dtype)
    values_sort=np.sort(toSort, order='sim')
    values_sort=np.flip(values_sort)

    top_n_imgs=[]
    for count in range(0,top_n):
        temp=values_sort[count]
        top_n_imgs.append(temp[1])

    top_n_imgs_array=np.asarray(top_n_imgs)
    return top_n_imgs_array








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
    project1=[]
    project2=[]
    x1_mean0=x1-mean_face
    x2_mean0=x2-mean_face
    for u_i in u.T:
        project1.append(np.inner(u_i,x1_mean0))
        project2.append(np.inner(u_i, x2_mean0))
    project1=np.asarray(project1)
    project2 = np.asarray(project2)
    interpolate_projects=np.linspace(project1,project2,n,endpoint=True)

    interpolate_imgs=[]
    for interpolate_project in interpolate_projects:
        interpolate_imgs.append(reconstruct_image(interpolate_project, mean_face, u))
    interpolate_imgs=np.asarray(interpolate_imgs)
    return interpolate_imgs