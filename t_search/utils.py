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

GPSolverStatus = Literal["INIT", "MAX_GEN", "MAX_EVAL", "MAX_ROOT_EVAL", "SOLVED"]

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