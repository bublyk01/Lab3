import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=20, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=50, axis=1)

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

Sigma_scipy = np.diag(sigma_scipy)

all_user_predicted_ratings = np.dot(np.dot(U_scipy, Sigma_scipy), VT_scipy) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)

original_nan_mask = ratings_matrix.isna()

preds_df_with_nans = preds_df.where(original_nan_mask, np.nan)


def get_top_10_recommendations(user_id, preds_df, original_ratings_df, num_recommendations=10):
    user_row_number = user_id - 1
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)

    user_data = original_ratings_df[original_ratings_df.userId == user_id]

    print("User's sorted predictions:")
    print(sorted_user_predictions.head(15))

    print("\nUser's full data:")
    print(user_data.head())

    recommendations = pd.DataFrame(sorted_user_predictions).reset_index()
    recommendations.columns = ['movieId', 'Predictions']
    recommendations = recommendations[~recommendations['movieId'].isin(user_data['movieId'])]
    recommendations = recommendations.sort_values('Predictions', ascending=False).iloc[:num_recommendations, :]

    return recommendations


user_id = 1
top_10_recommendations = get_top_10_recommendations(user_id, preds_df_with_nans, df)

print("\nMovie recommendations for ID {}:".format(user_id))
print(top_10_recommendations[['movieId']])
