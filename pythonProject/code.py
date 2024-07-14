import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

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

    return U, Sigma, V.T


U_custom, Sigma_custom, VT_custom = custom_svd(R_demeaned)

k = 3
U_scipy, sigma_scipy, VT_scipy = svds(R_demeaned, k=k)
Sigma_scipy = np.zeros((R_demeaned.shape[0], R_demeaned.shape[1]), dtype=float)
np.fill_diagonal(Sigma_scipy[:k, :k], sigma_scipy)

print("Custom SVD - U Matrix:")
print(U_custom)
print("\nCustom SVD - Sigma Matrix:")
print(Sigma_custom)
print("\nCustom SVD - VT Matrix:")
print(VT_custom)

print("\nScipy SVD - U Matrix:")
print(U_scipy)
print("\nScipy SVD - Sigma Matrix:")
print(Sigma_scipy)
print("\nScipy SVD - VT Matrix:")
print(VT_scipy)

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')

num_users = min(20, U_scipy.shape[0])
xs_scipy = U_scipy[:num_users, 0]
ys_scipy = U_scipy[:num_users, 1]
zs_scipy = U_scipy[:num_users, 2]

ax1.scatter(xs_scipy, ys_scipy, zs_scipy, c=range(num_users), cmap='viridis')
ax1.set_title('Users (Scipy SVD)')

ax2 = fig.add_subplot(122, projection='3d')

num_movies = min(20, VT_scipy.shape[1])
xs_movies = VT_scipy[0, :num_movies]
ys_movies = VT_scipy[1, :num_movies]
zs_movies = VT_scipy[2, :num_movies]

ax2.scatter(xs_movies, ys_movies, zs_movies, c=range(num_movies), cmap='viridis')
ax2.set_title('Movies (Scipy SVD)')

plt.show()
