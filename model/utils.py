import numpy as np
import math
import torch
from torchvision import transforms

IMAGE_SHAPE = (224, 224)
MACHINE_EPS = np.finfo(np.float32).eps

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SHAPE),
    transforms.ToTensor()
])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def top_k_indices(model, x, k):
    assert len(x.shape) == 3, "Only accept one image."
    x = x.unsqueeze(0) 

    with torch.no_grad():
        output = model(x)[0]
    return torch.topk(output, k).indices
    

def pairwise_distances(x):
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1).astype(np.float64)

def probs_to_entropy(probs, exclude_id=None):
    if exclude_id is not None:
        probs = np.delete(probs, exclude_id)

    probs = probs[probs != 0]
    return -np.inner(probs, np.log(probs))

def squared_dist_to_gaussian_conditional_prob(squared_dist_matrix, tgt_ppl):
    max_steps = 100
    tol = 1e-5

    n_samples = squared_dist_matrix.shape[0]
    tgt_entropy = math.log(tgt_ppl)
    
    con_matrix = np.zeros_like(squared_dist_matrix)

    # loop for each samples
    for i in range(n_samples):
        beta_min = None
        beta_max = None
        beta = 1.0 # beta = 1/(2 \sigma^2)

        for _ in range(max_steps):
            # compute p(*|i) and H(p(*|i) )
            for j in range(n_samples):
                if i != j:
                    con_matrix[i, j] = math.exp(-squared_dist_matrix[i, j] * beta)
                    
            if not con_matrix[i, :].any():
                # when con_matrix[i, :] is all zeros, add eps
                for j in range(n_samples):
                    if i != j:
                        con_matrix[i, j] += MACHINE_EPS

            con_matrix[i, :] /= con_matrix[i, :].sum()

            entropy = probs_to_entropy(con_matrix[i, :], exclude_id=i)

            entropy_diff = entropy - tgt_entropy
            if math.fabs(entropy_diff) < tol:
                # break the loop if subtle difference
                break
            else:
                # binary search
                if entropy_diff > 0.0:
                    beta_min = beta
                else:
                    beta_max = beta
                    
                if beta_max is None:
                    beta *= 2
                elif beta_min is None:
                    beta /= 2
                else:
                    beta = (beta_min + beta_max) / 2

    return con_matrix

def conditional_prob_to_joint_prob(con_matrix):
    # See https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding#:~:text=Now%20define

    n_samples = con_matrix.shape[0]
    return (con_matrix + con_matrix.T) / (2*n_samples)

def kl_loss(P, Q):
    non_zero = (Q != 0.0) & (P != 0.0)
    P = P[non_zero]
    Q = Q[non_zero]
    return np.sum(P * np.log(P / Q))


def squared_dist_to_student_t_joint_prob(squared_dist_matrix, degrees_of_freedom):
    n_samples = squared_dist_matrix.shape[0]
    
    t_dist = (squared_dist_matrix / degrees_of_freedom) + 1.0
    t_dist **= (degrees_of_freedom + 1.0) / -2.0

    joint_matrix = np.identity(n_samples)
    joint_matrix = np.where(joint_matrix == 0, t_dist ,0)
    joint_matrix /= joint_matrix.sum()

    return joint_matrix



