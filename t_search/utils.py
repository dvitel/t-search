''' Set of useful objects and functions '''
from time import perf_counter
from typing import Callable, Sequence, Literal
import numpy as np
import torch

from t_search.syntax.term import Term, TermPos

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

# y = torch.tensor([[3.1, 5.2, 7.1, 9.2]])
# x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
# k, b = optimize_kb(x, y)
# print(k)  # Should be close to [[2.0]]


def pearson_corr(x, y):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    covariance = (x_centered * y_centered).sum()
    x_std = torch.sqrt((x_centered ** 2).sum())
    y_std = torch.sqrt((y_centered ** 2).sum())
    return covariance / (x_std * y_std + 1e-8)

def spearman_corr(x, y):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    x_ranked = rank(x.unsqueeze(0)).squeeze(0)
    y_ranked = rank(y.unsqueeze(0)).squeeze(0)
    return pearson_corr(x_ranked, y_ranked)
