import numpy as np


def custom_svd(matrix):
    # Compute A^T * A
    ATA = np.dot(matrix.T, matrix)

    eigenvalues, V = np.linalg.eigh(ATA)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    V = V[:, sorted_indices]

    singular_values = np.sqrt(eigenvalues)

    Sigma = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=float)
    np.fill_diagonal(Sigma, singular_values)

    U = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    for i in range(len(singular_values)):
        U[:, i] = np.dot(matrix, V[:, i]) / singular_values[i]

    reconstructed_matrix = np.dot(U, np.dot(Sigma, V.T))

    print("Original Matrix:")
    print(matrix)
    print("\nU Matrix:")
    print(U)
    print("\nSigma Matrix:")
    print(Sigma)
    print("\nVT Matrix:")
    print(V.T)
    print("\nReconstructed Matrix:")
    print(reconstructed_matrix)

    return U, Sigma, V.T


matrix = np.array([[1, 2], [3, 4], [5, 6]])
U, Sigma, VT = custom_svd(matrix)
