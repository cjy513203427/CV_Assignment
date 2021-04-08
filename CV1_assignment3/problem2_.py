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
        m = np.shape(features1)[1]
        n = np.shape(features2)[1]
        # p=128
        distances = np.zeros((n, m))
        for idx1, feature_2 in enumerate(features2.T):
            for idx2, feature_1 in enumerate(features1.T):
                error = feature_2 - feature_1
                element = np.linalg.norm(error) ** 2
                distances[idx1][idx2] = element
        return distances

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
        min = m if m < n else n
        axis = 0 if m < n else 1  # 0: NACH OBEN; 1:NACH LINKS
        pairs = np.empty((min, 4))
        min_idx = np.argmin(distances, axis=axis)
        for i in range(0, min):
            if axis == 0:  # m<n
                pairs[i] = np.concatenate((p1[i], p2[min_idx[i]]))
            else:  # n<m
                pairs[i] = np.concatenate((p1[min_idx[i]], p2[i]))
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
        n = np.shape(p1)[0]
        m = np.shape(p2)[0]
        rng = np.random.default_rng()
        idx1 = rng.choice(n, k, replace=False)
        # idx2 = rng.choice(m, k, replace=False)
        sample1 = p1[idx1, :]
        sample2 = p2[idx1, :]
        return sample1, sample2

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
        l = np.shape(points)[0]
        norms=np.linalg.norm(points, axis=1)
        s = 0.5 * np.amax(norms)
        # s = 0.5 * np.amax(np.absolute(points))
        t = np.sum(points, axis=0) / l
        t_x = t[0]
        t_y = t[1]
        T = np.array([[1 / s, 0, -t_x / s], [0, 1 / s, -t_y / s], [0, 0, 1]])

        ones = np.ones((l, 1))
        points_homo = np.concatenate((points, ones), axis=1)
        ps = (T @ (points_homo.T)).T
        return ps, T

    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices shoul be normalized so that the bottom right value equals 1.
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
        l = np.shape(p1)[0]
        A = []
        for i in np.arange(0, l):
            p1_x = p1[i][0]
            p1_y = p1[i][1]
            p2_x = p2[i][0]
            p2_y = p2[i][1]
            A_row1 = np.array([0, 0, 0, p1_x, p1_y, 1, -p1_x * p2_y, -p1_y * p2_y, -p2_y])
            A_row2 = np.array([-p1_x, -p1_y, -1, 0, 0, 0, p1_x * p2_x, p1_y * p2_x, p2_x])
            A.append(A_row1)
            A.append(A_row2)
        A_array = np.asarray(A)
        u, s, vh = np.linalg.svd(A_array)
        # vh or vh.T ??
        h = vh[:, -1]
        HC = np.reshape(h, (3, 3))
        HC_normalized = HC / HC[2][2]
        H = np.linalg.inv(T2) @ HC @ T1
        H_normalized = H / H[2][2]

        return H_normalized, HC_normalized

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
        l = np.shape(p)[0]
        # how to convert p to homogeneous?
        ones = np.ones((l, 1))
        p_homo = np.concatenate((p, ones), axis=1)
        points_temp = (H @ p_homo.T).T
        points = points_temp[:, [0, 1]] / points_temp[:, [-1]]
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
        # homo
        # l = np.shape(p1)[0]
        # ones = np.ones((l, 1))
        # p1_homo=np.concatenate((p1,ones), axis=1)
        # p2_homo = np.concatenate((p2, ones), axis=1)

        # part1_error=(H@p1_homo.T).T-p2_homo
        p1_trans=self.transform_pts(p1, H)
        part1_error = p1_trans - p2
        part1 = np.linalg.norm(part1_error, axis=1) ** 2

        # part2_error=p1_homo-(np.linalg.inv(H)@p2_homo.T).T
        p2_trans=self.transform_pts(p2, np.linalg.inv(H))
        part2_error = p1 - p2_trans
        part2 = np.linalg.norm(part2_error, axis=1) ** 2

        dist = (part1 + part2).flatten()
        return dist

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
        N = 0
        inliers_list = []
        for idx, valid in enumerate(dist < threshold):
            N += 1 if valid else 0
            if valid:
                inliers_list.append(pairs[idx])
        inliers = np.asarray(inliers_list)
        return N, inliers

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
        w_d = np.power(p, k)
        number_iters = np.ceil(np.log(1 - z) / np.log(1 - w_d))
        return int(number_iters)

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
        p1 = pairs[:, [0, 1]]
        p2 = pairs[:, [2, 3]]
        H_list = []
        N_list = []
        inliers_list = []
        for i in range(0, n_iters):
            sample1, sample2 = self.pick_samples(p1, p2, k)

            ps1, T1 = self.condition_points(sample1)
            ps2, T2 = self.condition_points(sample2)
            H, HC = self.compute_homography(ps1, ps2, T1, T2)
            dist = self.compute_homography_distance(H, p1, p2)
            N, inliers = self.find_inliers(pairs, dist, threshold)
            H_list.append(H)
            N_list.append(N)
            inliers_list.append(inliers)
        idx = np.argmax(np.asarray(N_list))
        max_inliers = N_list[idx]
        H = H_list[idx]
        inliers = inliers_list[idx]
        return H, max_inliers, inliers

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
        p1 = inliers[:, [0, 1]]
        p2 = inliers[:, [2, 3]]
        ps1, T1 = self.condition_points(p1)
        ps2, T2 = self.condition_points(p2)
        H, HC = self.compute_homography(ps1, ps2, T1, T2)
        return H