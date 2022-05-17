import math
import numpy as np
from utils import (
    pairwise_distances, 
    squared_dist_to_gaussian_conditional_prob, 
    conditional_prob_to_joint_prob, 
    squared_dist_to_student_t_joint_prob,
    kl_loss,
    seed_everything
)

class TSNE:
    """t-Distributed Stochastic Neighbor Embedding
    """

    # Aligned to sklearn.SKTSNE 
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(
        self, 
        n_components=2, 
        perplexity=30.0, 
        early_exaggeration=12.0, 
        learning_rate=200.0, 
        n_iter=2000, 
        n_iter_without_progress=300, 
        min_grad_norm=1e-07,
        init="random"
    ):
        """ The arguments are a subset of sklearn.SKTSNE.
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.init = init
        self.degrees_of_freedom = float(max(self.n_components - 1, 1))
        self.h = None
    
    
    def fit_transform(self, x):
        n_samples = x.shape[0]

        # check parameters
        if self.learning_rate == "auto":
            self.learning_rate = max(n_samples / self.early_exaggeration / 4, 50)
        
        # initialize
        self.h = (1e-4 * np.random.randn(n_samples, self.n_components)) # Algorithm 1, "Visualizing Data Using t-SNE"
        
        # compute the distance matrix for samples in high dimension (squared)
        squared_dist_matrix_high = pairwise_distances(x)**2

        # compute con_p_matrix[i, j] = p(j|i)
        con_p_matrix = squared_dist_to_gaussian_conditional_prob(squared_dist_matrix_high, self.perplexity)

        # compute joint_p_matrix[i, j] = p(i, j)
        joint_p_matrix = conditional_prob_to_joint_prob(con_p_matrix)
        
        # Part 1
        joint_p_matrix *= self.early_exaggeration # early exaggeration
        it = self.update(joint_p_matrix, momentum=0.5, n_iter=self.n_iter)

        # Part 2
        joint_p_matrix /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            self.update(joint_p_matrix, momentum=0.8, n_iter=self.n_iter-it-1)
        
        return self.h

    def update(self, joint_p_matrix, momentum, n_iter):
        update = np.zeros_like(self.h)
        best_loss = math.inf
        it = best_iter = 0
        for it in range(n_iter):
            check_convergence = (it + 1) % self._N_ITER_CHECK == 0

            # compute the distance matrix for samples in low dimension (squared)
            squared_dist_matrix_low = pairwise_distances(self.h)**2

            # compute joint_q_matrix[i, j] = q(i, j)
            joint_q_matrix = squared_dist_to_student_t_joint_prob(squared_dist_matrix_low, self.degrees_of_freedom)

            # compute kl loss and grad
            loss = kl_loss(joint_p_matrix, joint_q_matrix)
            # print(loss)
            grad = self.grad(joint_p_matrix, joint_q_matrix)
            grad_norm = np.linalg.norm(grad)

            # update
            update = momentum * update - self.learning_rate * grad
            self.h += update

            # break the loop when converge
            if check_convergence:
                if loss < best_loss:
                    best_loss = loss
                    best_iter = it
                elif it - best_iter > self.n_iter_without_progress:
                    break
                if grad_norm <= self.min_grad_norm:
                    break
        return it

    def grad(self, p, q):
        # See Sec 2.2, "Learning a Parametric Embedding by Preserving Local Structure"

        grad = np.zeros_like(self.h)

        term1 = p - q # term1[i, j] = p_ij - q_ij   term1.shape: (i, j)
        term2 = self.h[:,np.newaxis,:] - self.h[np.newaxis,:,:] # term2[i, j] = h_i - h_j   term2.shape: (i, j, k)
        
        term3 =(term2**2).sum(axis=2) # term3[i, j] = ||h_i - h_j||^2   term3.shape: (i, j)
        term3 = (1 + (term3 / self.degrees_of_freedom))**(- (self.degrees_of_freedom + 1 ) / 2)

        grad = ((2 * self.degrees_of_freedom + 2) / self.degrees_of_freedom ) *  np.einsum('ij,ijk,ij->ik', term1, term2, term3)
        return grad