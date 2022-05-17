import numpy as np

class PCA:
    def __init__(self, n_componets=2):
        self.n_componets = n_componets
        self.pc_vectors = None
        self.x_mean = None
    
    def fit(self, x):
        x = x.copy()
        n_samples, n_features = x.shape
        assert self.n_componets <= n_features, f"{self.n_componets} <= {n_features}"

        x_mean = x.mean(axis=0)
        x = x - x_mean
        cov = (1.0 / n_samples) * np.dot(x.T, x)
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        eigen_vectors = eigen_vectors[:, np.argsort(-np.abs(eigen_values))]
        pc_vectors = eigen_vectors[:, :self.n_componets]

        self.pc_vectors = pc_vectors
        self.x_mean = x_mean

        return self

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def transform(self, x):
        assert (self.pc_vectors is not None) and (self.x_mean is not None)

        x = x - self.x_mean
        return np.matmul(x, self.pc_vectors) / np.diag(np.dot(self.pc_vectors.T, self.pc_vectors))