import numpy as np


def random_pd_matrix_generator(d=2):
    """Generates random symmetric positive definite matrices given dimension"""
    temp = np.random.randn(d, d)
    U = np.linalg.eig(temp + temp.T)[1]
    lambda_ = 1 + np.random.rand(d)
    pd_matrix = U @ np.diag(lambda_) @ U.T
    return pd_matrix
