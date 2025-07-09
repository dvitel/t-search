
''' Not finished - left for later 
We consider two version of comparison - 
1.  vectorize which would require checking all dimenstions
2.  early exit which would stop on first mismatch.
'''

from typing import Any, Callable, Optional, Sequence
import pandas as pd
import torch

# # Leads to constant recompilation - slow + require precise instalation of torch - got some lib errors
# @torch.compile
# def _all_close_dim(store: torch.Tensor, q: torch.Tensor, ids: torch.Tensor, dimId: int, rtol=1e-5, atol=1e-4) -> torch.Tensor:
#     ''' Check one dimenstion of store (1 column) and form the mask.
#         With fusion, we avoid unnecessary copying of tensors as tensors are eager. 
#         Store shape (N, dims), query q shape (dims), ids shape (K) to index K entries without copying        
#     '''
#     # return torch.isclose(store[ids, dimId], q[dimId], rtol=rtol, atol=atol)
#     return abs(store[ids, dimId] - q[dimId]) < rtol * abs(store[ids, dimId]) + atol

# def _all_close(store: torch.Tensor, q: torch.Tensor, rtol=1e-5, atol=1e-4) -> torch.Tensor:
#     ids = torch.arange(store.shape[0], device=store.device)
#     for dimId in range(q.shape[-1]):
#         mask = _all_close_dim(store, q, ids, dimId, rtol=rtol, atol=atol)
#         ids = ids[mask]
#         if ids.numel() == 0:
#             break
#     return ids

def __pred_ids(pred:  Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
               store: torch.Tensor, args: Sequence[Any], from_dim_id: int, ids: torch.Tensor, 
                    dim_delta = 1, permute_dim_id: Callable[[int], int] = lambda x: x) -> list[int]:
    ''' Utility, called by _pred
    '''
    cur_ids = ids
    cur_store = store[cur_ids]
    dim_id = from_dim_id
    while dim_id < store.shape[-1]:
        did = permute_dim_id(dim_id)
        did_end = did + dim_delta
        arg_slice = [(a[...,did:did_end] if torch.is_tensor(a) else a) for a in args ]
        local_mask = pred(cur_store[:, did:did_end], *arg_slice)
        num_matches = local_mask.sum().item()
        if num_matches == 0:
            return []
        if num_matches < len(cur_ids):  # at least one was removed
            cur_ids = cur_ids[local_mask]
            cur_store = store[cur_ids] #copying 
        dim_id += dim_delta
    return cur_ids.tolist()  # return list of ids of matches

def _pred(pred: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            store: torch.Tensor, *args: Any,
            ids_threshold = 64, dim_delta = 64,
            permute_dim_id: Callable[[int], int] = lambda x: x) -> list[int]:
    ''' Store shape (N, dims), query q shape (dims), returns ids of match 
        Note: store could be huge, we would like to avoid copying big chunks of it, indexing by collection leads to copying

        Experiment (test_early_exit_perf) give dim_delta = 64 as good choice.
        ids_threshold > 0 is benefiftial. It should be greater than E[number of entries with E[average common prefix]],
        therefore depends on store statistics. If it is expected that the store will not have many similar entries, default 64 is fine. 

        permute_dim_id could speedup the search by selecting first most discriminating dimensions, but this also requires storage statistics,
        It makes sense to have dims_stats of shape (dims) that tracks running variances
    '''
    mask = torch.ones(store.shape[0], dtype=torch.bool, device=store.device)  # mask of matches
    dim_id = 0
    while dim_id < store.shape[-1]:
        did = permute_dim_id(dim_id)  # get current dimension id
        did_end = did + dim_delta # in case if dim delta > 1, permutation shoud permute regions of size dim_delta
        arg_slice = [(a[...,did:did_end] if torch.is_tensor(a) else a) for a in args ]
        mask &= pred(store[:, did:did_end], *arg_slice)  # mask of matches
        num_matches = mask.sum().item()
        if num_matches == 0:
            return []
        if num_matches <= ids_threshold: # we can resort to copying from store to avoid unnecessary comparisons
            ids = torch.where(mask)[0]
            del mask
            return __pred_ids(pred, store, args, dim_id + dim_delta, ids,
                                   dim_delta = dim_delta, permute_dim_id = permute_dim_id)
        dim_id += dim_delta
    res = torch.where(mask)[0].tolist()  # return list of ids of matches
    return res


def close_pred(store, q, rtol=1e-5, atol=1e-4):
    return torch.isclose(store, q, rtol=rtol, atol=atol).all(dim=-1)

def range_pred(store, qmin, qmax):
    return ((store >= qmin) & (store <= qmax)).all(dim=tuple(range(1, store.ndim)))

def test_early_exit_perf(store_size = 500_000, dims = 1024, device = 'cpu', verbose=True):
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns

    dim_deltas = [1, 4, 16, 64, 256, 512]
    ids_thresholds = [0, 1000, 250_000, 300_000]
    common_prefix = [0, 1, 4, 16, 64, 256, 512, 768, 1000]
    params = [dict(dim_delta= dim_delta, ids_threshold=ids_threshold) for dim_delta in dim_deltas for ids_threshold in ids_thresholds]
    for cp_id, cp in enumerate(common_prefix):        
        store = torch.bernoulli(torch.full((store_size, dims), 0.5, device=device, dtype=torch.float16))
        rand_row_ids = torch.randint(0, store_size, (1,), device=device)
        rand_row = store[rand_row_ids]
        if cp > 0:
            store[0:250_000,0:cp] = rand_row[0,0:cp]  # make sure that first 512 dimensions are equal to the row we search for
        res = []
        for p in params:
            start_time = time.time()
            found_ids = _pred(close_pred, store, rand_row[0], **p) # search for a row
            assert found_ids[0] == rand_row_ids[0], "Early exit failed to find the row"
            end_time = time.time()
            duration = end_time - start_time
            print(f"Configuration: {p}")
            print(f"Execution Time: {duration:.6f} seconds")    
            res.append({**p, 'duration': duration})
        df = pd.DataFrame(res)
        heatmap = df.pivot(index = "dim_delta", columns = "ids_threshold", values = "duration")
        heatmap = heatmap.iloc[::-1]
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap, annot=32, 64, 128, 256, 512, 1024True, fmt=".2f", cmap="viridis_r", cbar=False, vmin=0, vmax=1)
        plt.title(f"Duration, store {store_size}, {dims}, {cp}")
        plt.xlabel("ids_threshold")
        plt.ylabel("dim_delta")
        plt.tight_layout()
        plt.savefig(f"data/early_exit/{cp_id:03}.png")
        plt.clf()
        del store
    pass

test_early_exit_perf(device="cuda")
pass


# def __eq_v(t1: torch.Tensor, t2: torch.Tensor) -> bool:
#     return torch.equal(t1, t2)

# def __eq_early_exit(t1: torch.Tensor, t2: torch.Tensor) -> bool:
#     # assert t1.shape[-1] == t2.shape[-1], "Number of dimensions should be equal"
#     for dim_id in range(t1.shape[-1]):
#         if not torch.equal(t1[..., dim_id], t2[..., dim_id]):
#             return False
#     return True

# def __le_v(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
#     return torch.all(t1 <= t2, dim=-1)

# def __le_early_exit(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
#     """ Early exit by last dimension """
#     assert t1.shape[-1] == t2.shape[-1], "Number of dimensions should be equal"
#     dims = t1.shape[-1]
#     if len(t1.shape) == 1 and len(t2.shape) == 1: # comparison to one vector 
#         t1 = t1.unsqueeze(0)
#     if len(t2.shape) == 1:
#         mask = torch.ones(t1.shape[:-1], dtype=torch.bool)
#         t1v = t1.view(-1, t1.shape[-1])  
#         maskv = mask.view(-1)
#         cur_mask = maskv
#         for dim_id in range(t2.shape[-1]):
#             cur_mask = t1v[cur_mask, dim_id] <= t2[dim_id]
#             t1v = t1v[cur_mask]
#             if t1v.numel() == 0:
#                 break
#     elif len(t1.shape) == 1:
#         mask = torch.ones(t2.shape[:-1], dtype=torch.bool)
#         t2v = t2.view(-1, t2.shape[-1])  
#         maskv = mask.view(-1)
#         cur_mask = maskv
#         for dim_id in range(t1.shape[-1]):
#             cur_mask = t1[dim_id] <= t2v[cur_mask, dim_id]
#             t2v = t2v[cur_mask]
#             if t2v.numel() == 0:
#                 break
#     else:
#         mask = torch.ones(t1.shape[:-1], dtype=torch.bool)
#         t1v = t1.view(-1, t1.shape[-1])  
#         t2v = t2.view(-1, t2.shape[-1])  
#         maskv = mask.view(-1)
#         cur_mask = maskv
#         for dim_id in range(t2.shape[-1]):
#             cur_mask = t1v[cur_mask, dim_id] <= t2v[cur_mask, dim_id]
#             t2v = t2v[cur_mask]
#             if t2v.numel() == 0:
#                 break
#     return mask

# def running_update(tensor: torch.Tensor, mask: torch.Tensor, update: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
#                     dim_permutation: Optional[torch.Tensor] = None) -> None:
#     ''' Runs through all dimensions in usual order or by dim_permutation (shape (dims))
#         On each step (dim_id), updates mask view with tensor view.
#         Param tensor shape is (N, dims), mask shape i (N, K)
#         Func update params: dim_id, mask_view, tensor_view, where mask_view, tensor_view not just view of dim_id, 
#             but also previously filtered by filters produced by previous infovations of func update (return value)
#             mask_view shape (L <= N, K), tensor_view shape (L <= N, 1), returns next iteration filter of shape (S <= N)
#         The result is in mask, as mask is gradually updated by updater.
#     '''
#     cur_mask = mask
#     cur_tensor = tensor
#     for dim_id in (dim_permutation or range(tensor.shape[-1])):
#         cur_filter_mask = update(dim_id, cur_mask, cur_tensor[:, dim_id])
#         # cur_mask.bitwise_and_(update(dim_id, cur_tensors[cur_mask, dim_id]))
#         cur_mask = cur_mask[cur_filter_mask] # viewing only successful passes of predicate
#         cur_tensor = cur_tensor[cur_filter_mask]
#         if cur_mask.numel() == 0:
#             break
#     return mask

# def find_eq(tensors: torch.Tensor, t: torch.Tensor, rtol=1e-5, atol=1e-4) -> torch.Tensor:
#     ''' Find indices where rows matches t.
#         tensors shape (N, dims), t shape (K, dims), or (dims).
#         returns 0 1 mask of shape (N, K)
#     '''
#     if len(t.shape) == 1:  # t is a single vector
#         t = t.unsqueeze(0)
#     tq = t.unsqueeze(0)  # (1, K, dims)
#     mask = torch.ones(tensors.shape[0], t.shape[0], dtype=torch.bool, device=tensors.device) # (N, K)
#     def _update(dim_id: int, cur_mask: torch.Tensor, cur_tensors: torch.Tensor) -> torch.Tensor:
#         cur_close = torch.isclose(cur_tensors.unsqueeze(1), tq[...,dim_id], rtol=rtol, atol=atol) # (L, K)
#         cur_mask.bitwise_and_(cur_close)
#         return torch.any(cur_close, dim=1) # (L)
#     running_update(tensors, mask, _update)
#     return mask

# def find_in(tensors: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
#     ''' Find indices where rows  are in between of tmin and tmax, tmin <= row <= tmax.
#         tensors shape (N, dims), tmin and tmax shape (dims) or (K, dims).
#         Result is of shape (N, K) mask 
#     '''
#     if len(tmin.shape) == 1:  # t is a single vector
#         tmin = tmin.unsqueeze(0)
#     if len(tmax.shape) == 1:  # t is a single vector
#         tmax = tmax.unsqueeze(0)
#     tminq = tmin.unsqueeze(0)  # (1, K, dims)
#     tmaxq = tmax.unsqueeze(0)  # (1, K, dims)
#     mask = torch.ones(tensors.shape[0], tmin.shape[0], dtype=torch.bool, device=tensors.device) # (N, K)
#     def _update(dim_id: int, cur_mask: torch.Tensor, cur_tensors: torch.Tensor) -> torch.Tensor:
#         cur_above = (cur_tensors >= tminq[..., dim_id])
#         cur_below = (cur_tensors <= tmaxq[..., dim_id])
#         cur_in = cur_above & cur_below
#         cur_mask.bitwise_and_(cur_in)
#         return torch.any(cur_in, dim=1) # (L)    
#     running_update(tensors, mask, _update)
#     return mask 
