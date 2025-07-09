
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

# TODO: study performance of batched versions: store vs qs (k, dims)

def __pred_ids(pred:  Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
               store: torch.Tensor, args: Sequence[Any], iter_id: int, num_iters: int, ids: torch.Tensor, 
                    dim_delta = 1, permute_dim_id: Callable[[int], int] = lambda x: x) -> list[int]:
    ''' Utility, called by _pred
    '''
    cur_ids = ids
    cur_store = store[cur_ids]
    while iter_id < num_iters:
        did = permute_dim_id(iter_id)
        did_end = did + dim_delta
        arg_slice = [(a[...,did:did_end] if torch.is_tensor(a) else a) for a in args ]
        local_mask = pred(cur_store[:, did:did_end], *arg_slice)
        num_matches = local_mask.sum().item()
        if num_matches == 0:
            return []
        if num_matches < len(cur_ids):  # at least one was removed
            cur_ids = cur_ids[local_mask]
            cur_store = store[cur_ids] #copying 
        iter_id += 1
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
    iter_id = 0
    num_iters = (store.shape[-1] + dim_delta - 1) // dim_delta  # number of iterations
    while iter_id < num_iters:
        did = permute_dim_id(iter_id)  # get current dimension id
        did_end = did + dim_delta # in case if dim delta > 1, permutation shoud permute regions of size dim_delta
        arg_slice = [(a[...,did:did_end] if torch.is_tensor(a) else a) for a in args ]
        mask &= pred(store[:, did:did_end], *arg_slice)  # mask of matches
        num_matches = mask.sum().item()
        if num_matches == 0:
            return []
        if num_matches <= ids_threshold: # we can resort to copying from store to avoid unnecessary comparisons
            ids = torch.where(mask)[0]
            del mask
            return __pred_ids(pred, store, args, iter_id + 1, num_iters, ids,
                                   dim_delta = dim_delta, permute_dim_id = permute_dim_id)
        iter_id += 1
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
        sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="viridis_r", cbar=False, vmin=0, vmax=1)
        plt.title(f"Duration, store {store_size}, {dims}, {cp}")
        plt.xlabel("ids_threshold")
        plt.ylabel("dim_delta")
        plt.tight_layout()
        plt.savefig(f"data/early_exit/{cp_id:03}.png")
        plt.clf()
        del store
    pass

if __name__ == "__main__":
    test_early_exit_perf(device="cuda")
    pass