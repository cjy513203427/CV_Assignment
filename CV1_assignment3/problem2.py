import numpy as np


class Problem2:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.

        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image

        Returns:
            distances: (n, m) numpy array, pairwise distances
        """
        #
        # You code here
        #
        from scipy.spatial.distance import cdist

        dis = cdist(features2.transpose(), features1.transpose(), metric='euclidean')

        return dis**2


    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.

        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix

        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """
        
        #
        # You code here
        #
        
        m = np.shape(p1)[0]
        n = np.shape(p2)[0]
        x = min(m,n)
        reshaped_p1 = p1[:x,]
        reshaped_p2 = p2[:x,]
        pairs = np.hstack((reshaped_p1,reshaped_p2))

        return pairs

    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.

        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select

        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """
        
        #
        # You code here
        #
        
        # 怎么接收参数p1和p2???

        random_p1 = np.arange(p1.shape[0])
        np.random.shuffle(random_p1)
        sample1 = p1[random_p1[0:k]]

        random_p2 = np.arange(p2.shape[0])
        np.random.shuffle(random_p2)
        sample2 = p2[random_p2[0:k]]

        return sample1,sample2





    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """

        #
        # You code here
        #

        normalized_points = (points - np.mean(points))/(0.5*np.max(np.abs(points)))
        homo_nor_points = np.concatenate((normalized_points, np.ones((normalized_points.shape[0], 1))), axis=1)

        # Vorlesung_07 Seite_14
        s_list = []
        for hnp in homo_nor_points:
            s_max = 0.5*np.linalg.norm(hnp)
            s_list.append(s_max)
        s = max(s_list)

        t = np.mean(homo_nor_points)
        T = np.array([[1/s,0,-t/s],
                    [0,1/s,-t/s],
                    [0,0,1]])

        return homo_nor_points,T



    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices should be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.

        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2
        
        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """

        #
        # You code here
        #

        p = np.vstack((p1,p2))
        u, s, vh = np.linalg.svd(p, full_matrices=True)
        HC_1 = np.dot(vh,T1)
        HC = np.dot(HC_1,T2)

        vh_nor = vh/vh[-1,-1]
        HC_nor = HC/HC[-1,-1]
        return vh_nor, HC_nor



    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """

        #
        # You code here
        #
        
        # transforms from homogeneous to cartesian coordinates
        res_1 = H[:-1,:]/H[-1,:]
        # rank 2 matrix
        res_2 = res_1[:,:-1]/res_1[:,-1]
        points = np.dot(p,res_2)

        return points




    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """
        #
        # You code here
        #
        dist_list_p1 = []
        for p in p1:
            dist = (np.linalg.norm(np.dot(H,p[0])-p[1]))**2 + (np.linalg.norm(p[0] - np.dot(np.linalg.inv(H),p[1])))**2
            dist_list_p1.append(dist)
        
        dist_list_p2 = []
        for p in p1:
            dist = (np.linalg.norm(np.dot(H,p[0])-p[1]))**2 + (np.linalg.norm(p[0] - np.dot(np.linalg.inv(H),p[1])))**2
            dist_list_p2.append(dist)
        
        dist_list = dist_list_p1 + dist_list_p2
        dist_arr = np.asarray(dist_list)

        return dist_arr[::2]
        




    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance. 

        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold
        
        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
        #
        # You code here
        #

        inliers_onedimension = pairs[pairs>threshold**2]
        # inliers = np.reshape(inliers_onedimension,(-1,4))

        return np.shape(inliers_onedimension)[0], pairs




    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """
        #
        # You code here
        #

        # Vorlesung_07 Seite_19 Formel
        min_iter = np.log(1 - z)/np.log(1 - p)

        return min_iter



    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold
        
        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """
        #
        # You code here
        #
        
        # number of inliers
        noi_list = []
        inliers_list = []
        H_list = []
        i = 0
        while i<n_iters:
            # randomly draw a sample of k corresponding point pairs
            sample1, sample2 = self.pick_samples(pairs[:,0:2], pairs[:,2:], k)
            sample1_nor,sample1_T = self.condition_points(sample1)
            sample2_nor,sample2_T = self.condition_points(sample2)
            # estimate the corresponding homography using homography function
            H,HC = self.compute_homography(sample1_nor,sample2_nor,sample1_T,sample2_T)
            # evaluate the homography by means of the homography distance specified above
            dist = self.compute_homography_distance(H,sample1,sample2)
            N, inliers = self.find_inliers(np.hstack((sample1,sample2)),dist,threshold)

            noi_list.append(N)
            inliers_list.append(inliers)
            H_list.append(H)
            i+=1
        
        noi = max(noi_list)
        noi_index = noi_list.index(noi)
        noi_inliers = inliers_list[noi_index]
        noi_H = H_list[noi_index]

        return noi_H,noi,noi_inliers
        


    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
        #
        # You code here
        #
        s = 0.5*np.linalg.norm(inliers)
        t = np.mean(inliers)
        H = np.array([[1/s,0,-t/s],
                    [0,1/s,-t/s],
                    [0,0,1]])
        
        return H

