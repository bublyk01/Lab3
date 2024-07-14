import numpy as np


def perform_svd(matrix):
    U, sigma, VT = np.linalg.svd(matrix)

    Sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
    np.fill_diagonal(Sigma, sigma)

    reconstructed_matrix = np.dot(U, np.dot(Sigma, VT))

    print("Original Matrix:")
    print(matrix)
    print("\nU Matrix:")
    print(U)
    print("\nSigma Matrix:")
    print(Sigma)
    print("\nVT Matrix:")
    print(VT)
    print("\nReconstructed Matrix:")
    print(reconstructed_matrix)

    return U, Sigma, VT


matrix = np.array([[1, 2], [3, 4], [5, 6]])
U, Sigma, VT = perform_svd(matrix)
