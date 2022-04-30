import numpy as np

def pca_transform(X, n_componets=2):
    """Apply PCA transform to data

    Args:
        X (numpy.ndarray): Input data of shape (n_samples, n_features), where n_samples is the number of samples and n_features is the number of features.
        n_componets (int, optional): The estimated number of components. Defaults to 2.
    """
    n_samples, n_features = X.shape
    assert n_componets <= n_features, f"{n_componets} <= {n_features}"

    X_mean = X.mean(axis=0)
    X = X - X_mean
    cov = (1.0 / n_samples) * np.dot(X.T, X)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    eigen_vectors = eigen_vectors[:, np.argsort(-np.abs(eigen_values))]
    pc_vectors = eigen_vectors[:, :n_componets]
    return np.matmul(X, pc_vectors) / np.diag(np.dot(pc_vectors.T, pc_vectors))
    
if __name__ == "__main__":
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    print(pca_transform(X, n_componets=2))
    



