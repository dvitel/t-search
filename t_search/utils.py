''' Set of useful objects and functions '''
from time import perf_counter
from typing import Callable, Sequence, Literal
import numpy as np
import torch

GLOBAL_RNG = np.random.default_rng() 

def add_metrics(metrics: dict, **kwargs):
    for key, value in kwargs.items():
        if key not in metrics:
            metrics[key] = value            
        elif isinstance(metrics[key], list):
            metrics[key].extend(value)
        else:
            metrics[key] = metrics[key] + value   

def timed(fn: Callable) -> Callable:
    """Decorator to time function execution"""

    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            elapsed_time = round((perf_counter() - start_time) * 1000)
        return result, elapsed_time

    return wrapper

def stack_rows(tensors: Sequence[torch.Tensor], dims: int) -> torch.Tensor:
    if len(tensors) == 0:
        raise ValueError("No tensors to stack")
        # return torch.empty((0, dims), dtype=target.dtype, device=target.device)
    if tensors[0].ndim <= 1:
        res = torch.empty((len(tensors), dims), dtype=tensors[0].dtype, device=tensors[0].device)
        for i, ti in enumerate(tensors):
            res[i] = ti # assuming broadcastable
        return res  
    if tensors[0].ndim == 2:
        sz = (sum(t.shape[0] for t in tensors), dims)
        res = torch.empty(sz, dtype=tensors[0].dtype, device=tensors[0].device)
        cur_start = 0
        for ti in tensors:
            res[cur_start:cur_start + ti.shape[0]] = ti
            cur_start += ti.shape[0]
        return res  
    raise ValueError(f"Unsupported tensor shape: {tensors[0].shape}")

def sorted_by_fitness(seq: Sequence, fitness: torch.Tensor, max_num: int | None = None) -> list:
    sorted_ids = torch.argsort(fitness, dim=0)
    if max_num is None:
        selected_ids = sorted_ids.tolist()
    else:
        selected_ids = sorted_ids[:max_num].tolist()
    return [seq[i] for i in selected_ids]

GPSolverStatus = Literal["INIT", "MAX_GEN", "MAX_EVAL", "MAX_ROOT_EVAL", "DEADEND", "SOLVED"]

class EvSearchTermination(Exception):
    """Reaching maximum of evals, gens, ops etc"""

    def __init__(self, status: GPSolverStatus, *args):
        super().__init__(*args)
        self.status = status

def rank(x: torch.Tensor):
    ''' 
        x is tensor of N traces  of size D.
        Produces tensor of same shape with ranks (averaged for ties) accross D points.
    '''
    N, D = x.shape
    device = x.device

    sorter = x.argsort(dim=-1) # ids to build sort orders 
    x_sorted = torch.gather(x, dim=-1, index=sorter) # sort

    # compare each element to its neighbor. True (1) if they are different.
    shifted = torch.cat([torch.full((N, 1), -float('inf'), device=device), x_sorted[:, :-1]], dim=-1)
    is_diff = (x_sorted != shifted)

    # use cumulative sum to label each ties
    group_ids = is_diff.cumsum(dim=-1) # (N, D)

    # calculate the ordinal ranks (0, 1, 2... D-1)
    ordinal_ranks = torch.arange(D, device=device).float().expand(N, D)

    # compute the mean of ordinal ranks within each group
    group_sums = torch.zeros(N, D + 1, device=device).scatter_add_(1, group_ids, ordinal_ranks)
    group_counts = torch.zeros(N, D + 1, device=device).scatter_add_(1, group_ids, torch.ones_like(ordinal_ranks))
    
    avg_ranks_per_group = group_sums / group_counts
    
    # map the average ranks back to the positions
    sorted_avg_ranks = torch.gather(avg_ranks_per_group, dim=-1, index=group_ids)
    
    # unsort
    inv_sorter = sorter.argsort(dim=-1)
    final_ranks = torch.gather(sorted_avg_ranks, dim=-1, index=inv_sorter)
    
    return final_ranks


# test_rank = torch.tensor([[3.1, 1.3, 2.1, 2.0],
#                           [3.4, 2.5, 3.4, 3.4],
#                           [2.1, 3.0, 1.1, 1.0],
#                           [1.1, 1.1, 1.1, 1.1],
#                           [5.1, 4.1, 3.1, 1.1]])

# rank(test_rank)  # Expected ranks with ties handled appropriately


def metrics_serializer(obj):
    from t_search.syntax.term import Term, TermPos
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, Term):
        return str(obj)
    if isinstance(obj, TermPos):
        return f"{obj.term}@{obj.occur}"
    raise TypeError(f"Type {type(obj)} not serializable")

def unique_vector_ids(vectors: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6) -> torch.Tensor:
    duplicate_mask = torch.isclose(vectors.unsqueeze(1), vectors.unsqueeze(0), atol=atol, rtol=rtol).all(dim=-1)
    lower_tri = torch.tril(duplicate_mask, diagonal=-1)
    del duplicate_mask
    has_duplicate_before = lower_tri.any(dim=1)  # (n,) - True if vector i is duplicate of some j < i
    del lower_tri
    unique_mask = ~has_duplicate_before  # (n,) - True for unique vectors
    del has_duplicate_before
    unique_indices = torch.where(unique_mask)[0]  # Indices of unique vectors
    del unique_mask
    return unique_indices

def unique_vector_ids_seq(vectors: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6) -> list[int]:
    uniq_ids: list[int] = []
    for i, vector in enumerate(vectors):
        if i == 0:
            uniq_ids.append(0)
            continue
        prev_vectors = vectors[uniq_ids]
        is_duplicate = torch.isclose(vector.unsqueeze(0), prev_vectors, atol=atol, rtol=rtol).all(dim=-1).any()
        if not is_duplicate:
            uniq_ids.append(i)
    return uniq_ids

def unique_vector_ids_batched(vectors: torch.Tensor, 
                                batch_size: int = 1024, max_size: int | None = None,
                                atol: float = 1e-6, rtol: float = 1e-6) -> torch.Tensor:
    cur_indices = torch.arange(vectors.shape[0], device=vectors.device)
    vector_id_groups = []
    for start in range(0, cur_indices.shape[0], batch_size):
        end = min(start + batch_size, cur_indices.shape[0])
        vector_id_group = cur_indices[start:end]    
        vector_id_groups.append(vector_id_group)
    cur_vector_ids = torch.empty((0, ), dtype=vector_id_groups[0].dtype, device=vector_id_groups[0].device)
    for vector_id_group in vector_id_groups:
        new_vector_ids = torch.cat([cur_vector_ids, vector_id_group])
        del vector_id_group, cur_vector_ids
        cur_vectors = vectors[new_vector_ids]
        unique_id_ids = unique_vector_ids(cur_vectors, atol=atol, rtol=rtol)
        cur_vector_ids = new_vector_ids[unique_id_ids]
        if max_size is not None and len(cur_vector_ids) >= max_size:
            cur_vector_ids = cur_vector_ids[:max_size]
            break
        del cur_vectors, unique_id_ids, new_vector_ids
    return cur_vector_ids


def optimize_kb(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ''' Searches for such k, b that minimizes MSE between Y and k*X+b  
        X - (n, dims), y (k,dims), k, b - (n, k)
        Assumes that X and Y have nonzero variations
    '''

    Sx = X.sum(dim=-1) #(n,)
    Sy = Y.sum(dim=-1) #(k,)
    Sxx = (X * X).sum(dim=-1) #(n,)
    Sxy = (X.unsqueeze(1) * Y.unsqueeze(0)).sum(dim=-1) #(n, k)
    dims = X.shape[-1]
    n_Covar_xy = (dims * Sxy - Sx.unsqueeze(1) * Sy.unsqueeze(0)) #(n, k)
    n_Var_x = (dims * Sxx - Sx * Sx) #(n,)
    k = n_Covar_xy / n_Var_x.unsqueeze(1)
    fix_mask = torch.isnan(k) | torch.isinf(k)
    k[fix_mask] = 0.0
    b = (Sy.unsqueeze(0) - k * Sx.unsqueeze(1)) / dims
    return k, b

def ransac_all_pairs(
    X,           # (N, dims)
    Y,           # (K, dims)
    iters=256,
    threshold=0.1,
    min_inliers=4,
    sample_size=2,
    torch_gen: torch.Generator | None = None
):
    device = X.device

    N, dims = X.shape
    K = Y.shape[0]

    # ------------------------------------------------------------
    # 1. sample random indices along dims axis
    # ------------------------------------------------------------

    rand = torch.rand(iters, dims, device=device, generator=torch_gen)

    sample_idx = torch.topk(rand, k=sample_size, dim=1, largest=False).indices
    # shape: (iters, 2)

    # ------------------------------------------------------------
    # 2. gather samples
    # ------------------------------------------------------------

    Xs = X[:, sample_idx]   # (N, iters, 2)
    Ys = Y[:, sample_idx]   # (K, iters, 2)

    # rearrange
    # Xs = Xs.permute(0,1,2)   # (N, iters, 2)
    # Ys = Ys.permute(0,1,2)   # (K, iters, 2)

    # ------------------------------------------------------------
    # 3. compute model parameters
    # ------------------------------------------------------------

    X_mean = Xs.mean(dim=2)   # (N, iters)
    Y_mean = Ys.mean(dim=2)   # (K, iters)

    Xc = Xs - X_mean[:,:,None]
    Yc = Ys - Y_mean[:,:,None]

    numerator = (
        Xc[:,None,:,:] * Yc[None,:,:,:]
    ).sum(dim=3)

    denominator = (Xc**2).sum(dim=2)[:,None,:]

    k = numerator / denominator.clamp(min=1e-10)
    # shape: (N, K, iters)

    b = Y_mean[None,:,:] - k * X_mean[:,None,:]

    # ------------------------------------------------------------
    # 4. evaluate all dims
    # ------------------------------------------------------------

    Y_pred = (
        k[:,:,:,None] * X[:,None,None,:]
        + b[:,:,:,None]
    )

    errors = torch.abs(Y_pred - Y[None,:,None,:])

    inliers = errors < threshold

    inlier_count = inliers.sum(dim=3)

    # ------------------------------------------------------------
    # 5. compute MSE
    # ------------------------------------------------------------

    sq_errors = errors**2

    mse = sq_errors.sum(dim=3) / inlier_count.clamp(min=1)
    mse_clone = mse.clone()

    mse[inlier_count < min_inliers] = float('inf')

    # ------------------------------------------------------------
    # 6. select best iteration
    # ------------------------------------------------------------

    best_iter = torch.argmin(mse, dim=2)

    k_best = torch.gather(
        k, 2, best_iter[:,:,None]
    ).squeeze(2)

    b_best = torch.gather(
        b, 2, best_iter[:,:,None]
    ).squeeze(2)

    X_counts = torch.gather(
        inlier_count, 2, best_iter[:,:,None]
    ).squeeze(2)    

    X_mse = torch.gather(
        mse_clone, 2, best_iter[:,:,None]
    ).squeeze(2)

    del Xs, Ys, X_mean, Y_mean, Xc, Yc, numerator, denominator, k, b, Y_pred, errors, inliers, inlier_count, sq_errors, mse, best_iter, mse_clone
    return k_best, b_best, X_counts, X_mse

# y = torch.tensor([[3.1, 5.2, 7.1, 9.2]])
# x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
# k, b = optimize_kb(x, y)
# print(k)  # Should be close to [[2.0]]


# # reproducibility
# torch.manual_seed(0)

# # small test sizes
# N = 2
# K = 4
# dims = 5

# device = 'cpu'  # change to 'cuda' if available

# # ------------------------------------------------------------
# # create ground truth parameters
# # ------------------------------------------------------------

# true_k = torch.randn(N, K, device=device) * 2.0
# true_b = torch.randn(N, K, device=device)

# # ------------------------------------------------------------
# # create base signals
# # ------------------------------------------------------------

# # X = torch.randn(N, dims, device=device)
# X = torch.tensor([[1.,2.,3.,4.,5.], [2., 2., 2., 2.,2.]])
# Y = torch.tensor([[2., 4., 6., 8., 10.], [4., 4., 4., 4., 4.], [6., 12., 18., 24., 30.], [8., 16., 24., 32., 40.]])

# generate Y from X using true k,b for first X only
# then mix to create general case
# Y = torch.empty(K, dims, device=device)

# for k_idx in range(K):
#     base_x = X[k_idx % N]
#     Y[k_idx] = true_k[k_idx % N, k_idx] * base_x + true_b[k_idx % N, k_idx]

# ------------------------------------------------------------
# add noise
# ------------------------------------------------------------

# Y += torch.randn_like(Y) * 0.001

# ------------------------------------------------------------
# inject outliers
# ------------------------------------------------------------

# outlier_mask = torch.rand(K, dims, device=device) < 0.2
# Y[outlier_mask] += torch.randn_like(Y[outlier_mask]) * 10.0

# ------------------------------------------------------------
# run RANSAC
# ------------------------------------------------------------

# k_est, b_est = ransac_all_pairs(
#     X, Y,
#     iters=2,
#     threshold=0.2,
#     min_inliers=3,
# )

# # ------------------------------------------------------------
# # evaluate error
# # ------------------------------------------------------------

# # compute prediction error
# Y_pred = k_est[:,:,None] * X[:,None,:] + b_est[:,:,None]

# errors = torch.mean((Y_pred - Y[None,:,:])**2, dim=2)

# print("Mean MSE across all pairs:", errors.mean().item())

# # print a few examples
# print("\nExample recovered parameters:\n")

# for i in range(min(3, N)):
#     for j in range(min(3, K)):
#         print(f"pair ({i},{j}):")
#         print(f"  k_est={k_est[i,j]:.3f}")
#         print(f"  b_est={b_est[i,j]:.3f}")
