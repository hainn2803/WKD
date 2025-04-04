import torch
import torch.nn as nn
import ot
import time
import numpy
import sys
import gc


class SinkhornAlgorithm(nn.Module):
    
    def __init__(self, epsilon=0.1, iterations=100, threshold=1e-9):
        super(SinkhornAlgorithm, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.threshold = threshold

    def _compute_matrix_H(self, u, v, cost_matrix):
        kernel = -cost_matrix + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel

    def forward(self, p, q, cost_matrix):

        u = torch.zeros_like(p)
        v = torch.zeros_like(q)

        for i in range(self.iterations):
            old_u = u
            old_v = v

            H = self._compute_matrix_H(u, v, cost_matrix)
            u = self.epsilon * (torch.log(p + 1e-8) - torch.logsumexp(H, dim=-1)) + u

            if H.ndim == 3:
                H = self._compute_matrix_H(u, v, cost_matrix).permute(0, 2, 1)
            else:
                H = self._compute_matrix_H(u, v, cost_matrix).permute(1, 0)

            v = self.epsilon * (torch.log(q + 1e-8) - torch.logsumexp(H, dim=-1)) + v

            diff = torch.sum(torch.abs(u - old_u), dim=-1) + torch.sum(torch.abs(v - old_v), dim=-1)
            mean_diff = torch.mean(diff)

            if mean_diff.item() < self.threshold:
                break

        K = self._compute_matrix_H(u, v, cost_matrix)
        pi = torch.exp(K)

        return pi




def generate_uniform_unit_sphere_projections(dim, num_projection=1000, dtype=torch.float32, device="cpu"):
    """
    Generate random uniform unit sphere projections with the same dtype as X and Y.
    """
    projection_matrix = torch.randn((num_projection, dim), dtype=dtype, device=device)
    return projection_matrix / torch.linalg.norm(projection_matrix, dim=1, keepdim=True)


def quantile_function(qs, cws, xs):
    cws, _ = torch.sort(cws, dim=0)
    qs, _ = torch.sort(qs, dim=0)
    num_dist = xs.shape[0]
    num_projections = xs.shape[-1]
    cws = cws.t().contiguous()
    qs = qs.t().contiguous()
    idx = torch.searchsorted(cws, qs).t()
    return torch.take_along_dim(input=xs, indices=idx.expand(num_projections, idx.shape[-1]).t().expand(num_dist, idx.shape[-1], num_projections), dim=-2)


def Wasserstein_Distance(X, Y, p=2, device="cpu"):
    """
    Compute the true Wasserstein distance. Can back propagate this function
    Computational complexity: O(n^3)
    :param X: M source samples. Has shape == (M, d)
    :param Y: N target samples. Has shape == (N, d)
    :param p: Wasserstein-p
    :return: Wasserstein distance (OT cost) == M * T. It is a number
    """

    assert X.shape[1] == Y.shape[1], "source and target must have the same"

    # cost matrix between source and target. Has shape == (M, N)
    M = ot.dist(x1=X, x2=Y, metric='sqeuclidean', p=p, w=None)

    num_supports_source = X.shape[0]
    num_supports_target = Y.shape[0]

    a = torch.full((num_supports_source,), 1.0 / num_supports_source, device=device)
    b = torch.full((num_supports_target,), 1.0 / num_supports_target, device=device)

    ws = ot.emd2(a=a,
                 b=b,
                 M=M,
                 processes=1,
                 numItermax=100000,
                 log=False,
                 return_matrix=False,
                 center_dual=True,
                 numThreads=1,
                 check_marginals=True)

    return ws



def Batch_Wasserstein_One_Dimension(X, Y, a=None, b=None, p=2, device="cuda"):
    """
    Compute the true Wasserstein distance in one-dimensional space in a batch.
    :param X: Source samples, shape (num_dists, M, d)
    :param Y: Target samples, shape (num_dists, N, d)
    :param p: Wasserstein-p order
    :return: Tensor of shape (num_dists, d) with Wasserstein distances
    """
    assert X.shape[-1] == Y.shape[-1], "Source and target must have the same dimension"
    assert X.shape[0] == Y.shape[0], "Source batch and target batch must have the same number of distributions"

    num_supports_source = X.shape[-2]
    num_supports_target = Y.shape[-2]

    if a is None and b is None:
        X_sorted, _ = torch.sort(X, dim=1)
        Y_sorted, _ = torch.sort(Y, dim=1)

        if num_supports_source == num_supports_target:
            diff_quantiles = torch.abs(X_sorted - Y_sorted)
            if p == 1:
                return torch.sum(diff_quantiles, dim=1) / num_supports_source
            else:
                return torch.pow(torch.sum(torch.pow(diff_quantiles, p), dim=1) / num_supports_source, 1/p)

        else:
            a_cum_weights = torch.linspace(1.0 / num_supports_source, 1.0, steps=num_supports_source, device=device)
            b_cum_weights = torch.linspace(1.0 / num_supports_target, 1.0, steps=num_supports_target, device=device)
            qs = torch.sort(torch.cat((a_cum_weights, b_cum_weights), 0), dim=0)[0]

            X_quantiles = quantile_function(qs, a_cum_weights, X_sorted)
            Y_quantiles = quantile_function(qs, b_cum_weights, Y_sorted)

            # del a_cum_weights, b_cum_weights
            # gc.collect()
            # torch.cuda.empty_cache()

            diff_quantiles = torch.abs(X_quantiles - Y_quantiles)

            qs_extended = torch.cat((torch.zeros(1, device=device), qs), dim=0)
            diff_qs = torch.clamp(qs_extended[1:] - qs_extended[:-1], min=1e-6)
            delta = diff_qs.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

            # del qs, qs_extended, diff_qs
            # gc.collect()
            # torch.cuda.empty_cache()

            return torch.pow(torch.sum(delta * torch.pow(diff_quantiles, p), dim=-2), 1/p) if p != 1 else torch.sum(delta * diff_quantiles, dim=-2)

    raise NotImplementedError("Weighted Wasserstein not implemented")



def Batch_Sliced_Wasserstein_Distance(X, Y, num_projections=1000, list_projection_vectors=None, p=2, device="cuda", chunk=1000):
    """
    Compute Sliced Wasserstein Distance in batch efficiently. Supports backpropagation.
    
    :param X: Batch of source measures, shape (num_dists, num_supports_source, d)
    :param Y: Batch of target measures, shape (num_dists, num_supports_target, d)
    :param num_projections: Number of projection directions
    :param list_projection_vectors: Optional precomputed projection vectors
    :param p: Wasserstein-p parameter (e.g., 1 or 2)
    :param device: Device to perform computation on (e.g., "cpu" or "cuda")
    :param chunk: Number of projections per chunk for memory management
    :return: Tensor of shape (A, B) with Sliced Wasserstein Distances
    """

    assert X.shape[-1] == Y.shape[-1], "Source and target must have the same dimension"
    assert X.shape[0] == Y.shape[0], "Source and target must have the same number of distributions"
    
    dims = X.shape[-1]
    num_dists = X.shape[0]

    if num_projections < chunk:
        chunk = num_projections
        chunk_num_projections = 1
    else:
        chunk_num_projections = num_projections // chunk

    sum_w_p = torch.zeros((num_dists), device=device)

    for i in range(chunk_num_projections):
        if list_projection_vectors is None:
            projection_vectors = generate_uniform_unit_sphere_projections(dim=dims, num_projection=chunk, device=device).detach()
        else:
            projection_vectors = list_projection_vectors[i].detach()

        projection_vectors = projection_vectors.to(torch.float16)

        X_projection = torch.matmul(X.to(torch.float16), projection_vectors.t()) # (batch_size, num_examples, num_projections)
        Y_projection = torch.matmul(Y.to(torch.float16), projection_vectors.t()) # (batch_size, num_examples, num_projections)

        # del projection_vectors
        # gc.collect()
        # torch.cuda.empty_cache()

        w_1d = Batch_Wasserstein_One_Dimension(X=X_projection, Y=Y_projection, p=p, device=device) # (batch_size, num_projections)

        # del X_projection, Y_projection
        # gc.collect()
        # torch.cuda.empty_cache()

        sum_w_p += torch.sum(torch.pow(w_1d, p), dim=-1)

        # del w_1d
        # gc.collect()
        # torch.cuda.empty_cache()

    mean_w_p = sum_w_p / num_projections
    return torch.pow(mean_w_p, 1/p) if p != 1 else mean_w_p


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)

    X = torch.randn(2, 10, 3, requires_grad=True, device="cuda")
    Y = torch.randn(2, 10, 3, requires_grad=False, device="cuda")

    sw = Batch_Sliced_Wasserstein_Distance(X, Y, num_projections=100, p=2, device="cuda")

    print(sw.shape)
    sw_dist = torch.mean(sw)

    sw_dist.backward()

    print(X.grad == None)  # Should not be None
    print(Y.grad == None)  # Should not be None
