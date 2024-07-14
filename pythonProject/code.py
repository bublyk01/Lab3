import numpy as np
import pandas as pd

file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)

average_mark = 2.5
ratings_matrix_filled = ratings_matrix.fillna(average_mark)

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


def custom_svd(matrix):
    # Compute A^T * A
    ATA = np.dot(matrix.T, matrix)

    eigenvalues, V = np.linalg.eigh(ATA)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    V = V[:, sorted_indices]

    singular_values = np.sqrt(np.maximum(eigenvalues, 0))

    Sigma = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=float)
    np.fill_diagonal(Sigma, singular_values)

    U = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    for i in range(len(singular_values)):
        U[:, i] = np.dot(matrix, V[:, i]) / singular_values[i] if singular_values[i] != 0 else 0

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


U, Sigma, VT = custom_svd(R_demeaned)
