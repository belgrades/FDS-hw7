import numpy as np


def euc(x, y):
    diff = x - y
    return np.sqrt(np.dot(diff.T, diff))


def alldist(X):
    """
    Calculates the Euclidean distance matrix between all pairs of rows in the NumPY array X.
    :param X: A n-by-n NumPy array where n is the number of rows of X.
    :return: Euclidean distance.
    """
    n = X.shape[0]
    D = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i + 1, n):
            D[i, j] = euc(X[i], X[j])
            D[j, i] = D[i, j]
    return D


# Generate the D-by-d random matrix used by Achlioptas' algorithm. The i-th column of the matrix
# specifies the weights with which the coordinates of a D-dimensional vector are combined to produce
# the i-th entry of its d-dimensional transform. Each entry is chosen independently and uniformly at
# random in {-1,1}
def achmat(D,d):
    return np.where((np.random.randn(D,d) < 0), -1, 1)


# Perform dimensionality reduction on a set of points. Takes in input a set of n points in
# R^D , in the form of an n-by-D matrix X, and reduces each point to have dimensionality
# d. It returns the n-by-d NumPy array containing the n reduced points as rows. Recall
# that the linear map that reduces point x is f(x) = d^−0.5 * X^T * A, where A is the matrix
# obtained with achmat(), and T means transpose.
def reduce(X, d):
    A = achmat(X.shape[1], d)
    return np.dot(X,A) / np.sqrt(d)


# Compute the distance distortion between all pairs of points, according to the distances
# in the matrices dm1 and dm2. Recall that the distance distortion between point i and
# point j is simply dm2(i,j)/dm1(i,j). The result is a NumPy array of length n(n − 1)/2
# containing, for each pair of points (i, j) with 1 <= i < j <= n, the ratio between their
# distance according to dm1 and their distance according to dm2. (Note that the distance
# between a point and itself is not considered).
def distortion(dm1, dm2):
    mat = dm1/dm2
    return mat[np.triu_indices(len(mat), k=1)]
