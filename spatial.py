''' 
Implementation of spatial indices for efficient approximate nearest neighbor search (in amortized sense).

Idea of spacial indices is to avoid full search. 
For vector x and all semantix X of size n, full search would
require O(n) comparisons of k vector values. 
'''

from dataclasses import dataclass
from functools import partial
from itertools import product
import math
from typing import Callable, Generator, Literal, Optional, Union
from matplotlib import pyplot as plt
import numpy as np
import torch

def get_by_ids(vectors: torch.Tensor, max_size: int, ids: None | int | tuple[int, int] | list[int]):
    ''' From a big store of vectors (n, dims) selects view or resort to advanced indecing 
        which leads to copying of values (materialization)
    '''
    if ids is None:
        return vectors[:max_size] # view
    if type(ids) is tuple:
        return vectors[ids[0]:ids[1]] # view by range
    if isinstance(ids, int):
        return vectors[ids[0]] # return a view by index 
    if len(ids) == 1:
        return vectors[ids[0]].unsqueeze(0) # also view
    return vectors[ids] # this will new tensor with values copied from mindices        

def find_pred(store: torch.Tensor, query: torch.Tensor, 
                pred: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                store_batch_size = 1024, query_batch_size = 256, dim_batch_size = 8) -> list[int]:
    ''' store shape (n, *k, dims), query shape (m, *k, dims), 
        computes mask (n, m) of True/False - x 'row' is close to y 'row' 
        returns list of row ids of store where query matches or -1 if no match
    '''
    mask = torch.ones(store.shape[0], query.shape[0], dtype=torch.bool, device=store.device)  # mask of matches
    for store_chunk_id, store_chunk in enumerate(torch.split(store, store_batch_size, dim=0)):        
        store_start = store_chunk_id * store_batch_size
        store_end = store_start + store_batch_size
        for query_chunk_id, query_chunk in enumerate(torch.split(query, query_batch_size, dim=0)):
            query_start = query_chunk_id * query_batch_size
            query_end = query_start + query_batch_size
            for store_dim_chunk, query_dim_chunk in \
                    zip(torch.split(store_chunk, dim_batch_size, dim=-1), 
                                torch.split(query_chunk, dim_batch_size, dim=-1)):
                # dim_start = dim_chunk_id * dim_batch_size
                mask_per_el = pred(store_dim_chunk.unsqueeze(1), query_dim_chunk.unsqueeze(0)) # (n, m, *k, dims)
                mask[store_start:store_end, 
                     query_start:query_end] &= mask_per_el.all(dim = tuple(range(2, store.ndim + 1))) # (n, m)
                del mask_per_el
                if not torch.any(mask[store_start:store_end, query_start:query_end]):
                    break
    store_ids, query_ids = torch.where(mask) # tuple row (ids1, ids2) where idsX has number of matches and indexes for dimension X
    ids = torch.full((mask.shape[-1],), -1, dtype = torch.int64, device=mask.device)
    ids[query_ids] = store_ids
    del mask
    ids_list = ids.tolist()
    del ids 
    return ids_list

def find_close(store: torch.Tensor, query: torch.Tensor, rtol=1e-5, atol=1e-4,
                    store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 64) -> list[int]:
    ''' store shape (n, *k, dims), query shape (m, *k, dims), 
        computes mask (n, m) of True/False - x 'row' is close to y 'row' 
        returns list of row ids of store where query matches or -1 if no match
    '''
    found_ids = find_pred(store, query, 
                          lambda s, q: torch.isclose(s, q, rtol=rtol, atol=atol), 
                          store_batch_size=store_batch_size, query_batch_size=query_batch_size, dim_batch_size=dim_batch_size)
    return found_ids

def find_equal(store: torch.Tensor, query: torch.Tensor,
               store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 64) -> list[int]:
    found_ids = find_pred(store, query, 
                          lambda s, q: s == q,
                          store_batch_size=store_batch_size, query_batch_size=query_batch_size, dim_batch_size=dim_batch_size)
    return found_ids

def find_in_range(store: torch.Tensor, query: torch.Tensor,
                    store_batch_size = 1024, dim_batch_size = 64) -> list[int]:
    ''' query shape is (2, *x.shape[1:]) - one per min and max 
        m - number of ranges to test. Usually m = 1, then 
    '''
    # # store_view = store.unsqueeze(1) # (n, 1, *k, dims)
    # # ymin = query[0].unsqueeze(0).unsqueeze(0) # (1, 1, *k, dims)
    # # ymax = query[1].unsqueeze(0).unsqueeze(0) # (1, 1, *k, dims)
    # mask_per_el = ((store >= query[0]) & (store <= query[1]))
    # mask = mask_per_el.all(dim = tuple(range(1, store.ndim)))
    # del mask_per_el
    # ids, = torch.where(mask)
    # del mask 
    mask = torch.ones(store.shape[0], dtype=torch.bool, device=store.device)  # mask of matches
    for store_chunk_id, store_chunk in enumerate(torch.split(store, store_batch_size, dim=0)):        
        store_start = store_chunk_id * store_batch_size
        store_end = store_start + store_batch_size
        for store_dim_chunk, query_dim_chunk in \
                zip(torch.split(store_chunk, dim_batch_size, dim=-1), 
                            torch.split(query, dim_batch_size, dim=-1)):
            mask_per_el = ((store_dim_chunk >= query_dim_chunk[0]) & (store_dim_chunk <= query_dim_chunk[1]))
            mask[store_start:store_end] &= mask_per_el.all(dim = tuple(range(1, store.ndim)))
            del mask_per_el
            if not torch.any(mask[store_start:store_end]):
                break
    ids, = torch.where(mask)
    del mask 
    return ids.tolist()

def get_mbr(vectors:torch.Tensor) -> torch.Tensor:
    if len(vectors.shape) == 1:  # single vector
        return torch.stack((vectors, vectors))
    mbr = torch.empty((2, vectors.shape[1]), dtype=vectors.dtype, device=vectors.device)
    mbr[0] = vectors.min(dim=0).values
    mbr[1] = vectors.max(dim=0).values    
    return mbr

def remap_ids(ids: None | tuple[int, int] | list[int], local_ids: list[int]) -> list[int]:
    if ids is None:
        return local_ids
    if isinstance(ids, tuple):
        found_ids = [-1 if local_id == -1 else (ids[0] + local_id) for local_id in local_ids]
        return found_ids
    # here ids is list
    found_ids = [-1 if local_id == -1 else ids[local_id] for local_id in local_ids ]
    return found_ids

def get_missing_ids(found_ids: list[int]):
    return [i for i, found_id in enumerate(found_ids) if found_id == -1]

def merge_ids(found_ids: list[int], new_ids: list[int]) -> list[int]:
    missing_id = 0 
    for i, found_id in enumerate(found_ids):
        if found_id == -1:
            found_ids[i] = new_ids[missing_id]
            missing_id += 1
    return found_ids

class StorageStats:     

    def __init__(self, storage: "VectorStorage"):
        ''' Storage statistics for vector storage.
            Computes running means, variances, min and max for each dimension.
            Useful to speedup comparisons and queries.
            batch_size: Number of vectors to process in one recompute call.
        '''
        self.storage = storage
        self.num_vectors: int = 0
        self.dim_means: Optional[torch.Tensor] = None 
        self.dim_variances: Optional[torch.Tensor] = None
        self.var_dim_permutation: Optional[torch.Tensor] = None 
        ''' Order of dimensions from most variative to least variative 
            Useful to shortcut comparisons. 
        '''
        self.dim_mins: Optional[torch.Tensor] = None
        self.dim_maxs: Optional[torch.Tensor] = None

    def recompute(self, *, dim_delta = 1) -> None:
        delayed_count = self.storage.cur_id - self.num_vectors
        if delayed_count <= 0:
            return
        batch = self.storage.get_vectors((self.num_vectors, self.num_vectors + delayed_count)) # latest nonprocessed
        n_batch = batch.size(0)
        mean_batch = batch.mean(dim=0)
        var_batch = batch.var(dim=0, unbiased=False)
        n_new = self.num_vectors + n_batch
        mean_new = (self.num_vectors * self.dim_means + n_batch * mean_batch) / n_new
        var_new = ((self.num_vectors * self.dim_variances + n_batch * var_batch) / n_new
                    + (self.num_vectors * n_batch * (self.dim_means - mean_batch)**2) / (n_new**2))
        self.num_vectors = n_new
        self.dim_means = mean_new
        self.dim_variances = var_new
        if dim_delta == 1:
            self.var_dim_permutation = torch.argsort(self.dim_variances, descending=True)
        else:
            dim_ids = torch.arange(0, batch.shape[-1], dim_delta, device=batch.device)
            self.var_dim_permutation = torch.argsort(self.dim_variances[dim_ids], descending=True) 
            self.var_dim_permutation *= dim_delta
        if self.dim_mins is None:
            self.dim_mins = batch.min(dim=0).values
            self.dim_maxs = batch.max(dim=0).values
        else:
            self.dim_mins = torch.minimum(self.dim_mins, batch.min(dim=0).values)
            self.dim_maxs = torch.maximum(self.dim_maxs, batch.max(dim=0).values)
        del batch, mean_batch, var_batch

class VectorStorage:
    ''' Interface for storage for indexing '''

    def __init__(self, capacity: int, dims: int, dtype = torch.float16,
                 rtol: float = 1e-5, atol: float = 1e-4, device: str = "cpu",
                 store_batch_size: int = 1024, query_batch_size: int = 128, dim_batch_size: int = 8):
                #  stats_batch_size: int = math.inf, dim_delta: int = 64):
        self.capacity = capacity 
        self.vectors = torch.empty((capacity, dims), dtype=dtype, device=device)
        self.cur_id = 0
        self.rtol = rtol
        self.atol = atol
        self.store_batch_size = store_batch_size
        self.query_batch_size = query_batch_size
        self.dim_batch_size = dim_batch_size
        # self.stats_batch_size = stats_batch_size
        # self.stats = StorageStats(self)

    def get_vectors(self, ids: None | int | list[int] | tuple[int, int]) -> torch.Tensor:
        ''' ids None --> whole storage view 
            ids int --> single vector view by id
            ids list --> tensor - copy of corresponding vectors 
        '''
        return get_by_ids(self.vectors, self.cur_id, ids)
    
    def alloc_vectors(self, vectors: torch.Tensor) -> list[int]:
        ''' Adds vectors (n, dims) to storage and returns new ids. '''
        self.vectors[self.cur_id:self.cur_id + vectors.shape[0]] = vectors
        vector_ids = list(range(self.cur_id, self.cur_id + vectors.shape[0]))
        self.cur_id += vectors.shape[0]
        # if self.cur_id - self.stats.num_vectors >= self.stats_batch_size:
        #     self.stats.recompute(dim_delta = self.dim_delta)
        return vector_ids
    
    # def _find(self, ids: None | tuple[int, int] | list[int], op, *args) -> list[int]:
        
    #     selection = self.get_vectors(ids)        
    #     permute_dim_id = lambda x:x*self.dim_delta 
    #     if self.stats and self.stats.var_dim_permutation:
    #         permute_dim_id = lambda x: self.stats.var_dim_permutation[x]
    #     found_ids = op(selection, *args, ids, dim_delta=self.dim_delta, 
    #                         permute_dim_id = permute_dim_id)
    #     return found_ids
    
    def find_close(self, ids: None | tuple[int, int] | list[int], q: torch.Tensor) -> list[int]:
        selection = self.get_vectors(ids)
        local_ids = find_close(selection, q, rtol=self.rtol, atol=self.atol,
                               store_batch_size=self.store_batch_size, 
                               query_batch_size=self.query_batch_size, 
                               dim_batch_size=self.dim_batch_size)
        del selection
        return remap_ids(ids, local_ids)
    
    def find_in_range(self, ids: None | tuple[int, int] | list[int], query: torch.Tensor) -> list[int]:
        selection = self.get_vectors(ids)
        local_ids = find_in_range(selection, query, store_batch_size=self.store_batch_size,
                                  dim_batch_size=self.dim_batch_size)
        del selection
        return remap_ids(ids, local_ids)
    
    def find_mbr(self, ids: None | tuple[int, int] | list[int]) -> Optional[torch.Tensor]:
        ''' Computes min and max tensors for given ids 
            Return shape (2, dims) with min in first row and max in second row.
        '''
        # if isinstance(ids, list) and len(ids) > 1: # running min max impl 
        #     res = torch.tensor(2, self.vectors.shape[1], dtype=self.vectors.dtype, device=self.vectors.device)
        #     res[0] = self.vectors[ids[0]]
        #     res[1] = self.vectors[ids[0]]
        #     for i in range(1, len(ids)):
        #         res[0] = torch.minimum(res[0], self.vectors[ids[i]])
        #         res[1] = torch.maximum(res[1], self.vectors[ids[i]])
        #     return res
        selection = self.get_vectors(ids)
        return get_mbr(selection)
        
# t1 = torch.tensor([
#     [0, 1, 0, 0],
#     [0, 1, 1, 0],
#     [1, 0, 0, 0],
# ])

# t [0, 1, 2, 1]



# res = torch.where(t1)
# pass

# t1 = torch.tensor([ [2,3,3,4,7,6], [2,2,4,4,7,6], [1,2,3,4,5,6], [2,2,3,4,5,6], [2,3,3,4,5,6], [1,2,3,4,5,6]])
# t2 = torch.tensor([[1,2,3,4,5,6], [2,2,4,4,7,6]])
# t3 = torch.tensor([1,2,3,4,5,6])
# t4 = torch.tensor([6,6,6,6,6,6])

# R1 = torch.tensor([ 
#     [   
#         [2,2,3,4,7,6], 
#         [2,3,4,4,7,6], 
#     ],
#     [
#         [1,2,3,4,5,6], 
#         [2,2,3,4,5,6], 
#     ], 
#     [
#         [1,2,3,4,5,6], 
#         [2,3,4,4,5,6]
#     ],
#     [
#         [1,2,3,4,5,6], 
#         [2,2,3,4,8,6], 
#     ],     
#     ])
# R2 = torch.tensor([
#     [1,2,3,4,5,6], 
#     [2,3,4,4,5,6]
# ])
# # R3 = torch.tensor([2,2,2,2,2,2])
# # R4 = torch.tensor([6,6,6,6,6,6])
# res = find_eq(t1, t4)

# # res = find_eq(R1, R2)
# res = find_in(t1, t3, t4)
# pass

class SpatialIndex(VectorStorage):
    ''' Defines the interface for spatial indices. 
        This is default implementation which isi heavy inefficient O(N), N - number of semantics.
    '''
    
    def query_points(self, points: torch.Tensor) -> list[int]:
        ''' O(n). Return id of vectors in index if present. Empty tensor otherwise. '''
        return self.find_close(None, points) # q here is one point among N points of all_vectors off shape (N, ..., dims)
    
    def query_range(self, qrange: torch.Tensor) -> list[int]:
        ''' O(n). Returns ids stored in the index, shape (N), N >= 0 is most cases.
            qrange[0] - mins, qrange[1] - maxs, both of shape (dims).
        '''
        return self.find_in_range(None, qrange)
    
    def _insert_distinct(self, unique_vectors: torch.Tensor) -> list[int]:
        found_ids = self.query_points(unique_vectors)
        missing = get_missing_ids(found_ids)
        if len(missing) > 0:
            missing_ids = self.alloc_vectors(unique_vectors[missing])
            found_ids = merge_ids(found_ids, missing_ids)
        return found_ids

    def insert(self, vectors: torch.Tensor) -> list[int]:
        ''' Insert vectors into index.
        '''
        unique_vectors, unique_indices = torch.unique(vectors, dim=0, return_inverse=True)
        found_ids = self._insert_distinct(unique_vectors)
        ids = [found_ids[unique_id] for unique_id in unique_indices.tolist()]
        del unique_vectors, unique_indices
        return ids

    # def query(self, q: torch.Tensor) -> list[int]:
    #     ''' Point and Range (rectangular) query.
    #         For point query, q has shape [dims], result has 0 or 1 element id depending on whether point is found.            
    #         For range query, q has shape [N, dims], 
    #             if N = 2, q is range per dimension, result is all points that are in the range.
    #             for N > 2, result depends on index, default behavior is to treat each ow as point and find
    #                        min max goting back to [2, dims] query.
    #     '''
    #     # assert 1 <= len(q.shape) <= 2, "Supporting only point and range queries with shapes (dims) or (N, dims)"
    #     if len(q.shape) == 1: # query point
    #         return self.query_points(q)
    #     else:
    #         if q.size(0) == 1:
    #             return self.query_points(q[0])
    #         if q.size(0) > 2:
    #             qmin = q.min(dim=0).values
    #             qmax = q.max(dim=0).values
    #             qrange = torch.stack((qmin, qmax), dim=0) # (2, dims)
    #         else:
    #             qrange = q
    #         return self.query_range(qrange)
        
class BinIndex(SpatialIndex):
    ''' Abstract index that is based on some binning schema. 
        Maps tensor (1, dims) to low dimensional usually discrete representation representation:
         1. Grid preserves dims but discretizes tensor 
            - preserves locality to a good degree
         2. RCos maps tensor to 2D space by R and cos distance to given target 
            - only partially preserves locality, need additional checks after selection
         3. Interaction - index that maps tensor to k compressed dims of 0-1 vectors
         4. Hash ? - maps tensor to 1D space by hash function but does not preserve any locality 
            - only useful for direct querying for equality, given epsilon allows to round before hashing
    '''
    
    def __init__(self, # epsilons: float | torch.Tensor = 1e-3, 
                max_bin_size: int = math.inf, 
                switch_to_all_cap = 0.5, **kwargs):
        ''' 
            epsilon: Size of the bin in each dimension (0 or 1 dim tensor)
            max_bin_size: Maximum number of elements in a bin, if set, 
                            grid resize (expensive) will be triggered with new epsilon that 
                            would satisfy this condition.
        '''
        super().__init__(**kwargs)
        # self.epsilons = epsilons
        self.max_bin_size = max_bin_size
        self.switch_to_all_cap = switch_to_all_cap
        self.bins: dict[tuple, list[int]] = {} # tuple is bin index

    def get_bin_index(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Should be defined by concrete index 
            vectors shape (n, dims)
        '''
        pass

    def on_rebuild(self, trigger_bin_id: tuple):
        ''' Index specific '''
        pass 
        
    def rebuild(self, trigger_bin_id: tuple):
        ''' Recreating bins and reindexing entries. O(n). Should not be frequent.
            Happens when trigger bin size exceeds max_bin_size.
        '''
        start_r_time = time.time()
        iters = 0
        while len(self.bins[trigger_bin_id]) >= self.max_bin_size:
            self.on_rebuild(trigger_bin_id)
            
            all_vectors = self.get_vectors(None) # get all 
            new_indices = self.get_bin_index(all_vectors)
            bin_ids = [tuple(row) for row in new_indices.tolist()]
            self.bins = {}
            for vector_id, bin_id in enumerate(bin_ids):
                self.bins.setdefault(bin_id, []).append(vector_id)
            del new_indices
            # prev_len = len(trigger_bin)
            trigger_bin_id = max(self.bins.keys(), key=lambda i: len(self.bins[i]))
            iters += 1
            # assert len(trigger_bin) < prev_len, "Size should decrease after rebuild"
        end_r_time = time.time()    
        print(f"\tRebuild: {end_r_time - start_r_time}. Sz: {len(self.bins[trigger_bin_id])} Iter: {iters}, Eps: {min(self.epsilons.tolist())}:{max(self.epsilons.tolist())}")     
        pass

    def _insert_distinct(self, unique_vectors: torch.Tensor) -> list[int]:
        ''' Insert unique_vectors (n, dims) and returns their ids '''
        bin_indices = self.get_bin_index(unique_vectors)
        group_by_bins = {}
        vector_bin_ids = {}
        for vector_id, bin_id in enumerate(bin_indices.tolist()):
            bid = tuple(bin_id)
            group_by_bins.setdefault(bid, []).append(vector_id)
            vector_bin_ids[vector_id] = bid
        del bin_indices
        if len(self.bins) == 0: 
            # if no bins, allocate all vectors
            self.bins = group_by_bins            
            found_ids = self.alloc_vectors(unique_vectors)
        else:
            trigger_bin_id = None
            all_bin_entries = [e for bin_index in self.bins.keys() for e in self.bins.get(bin_index, [])]
            if len(all_bin_entries) > self.switch_to_all_cap * self.cur_id:
                all_bin_entries = None            
            found_ids = self.find_close(all_bin_entries, unique_vectors) # recompute stats if needed
            missing_ids = get_missing_ids(found_ids)
            if len(missing_ids) > 0:
                new_ids = self.alloc_vectors(unique_vectors[missing_ids])
                found_ids = merge_ids(found_ids, new_ids)
                for vector_id in missing_ids:
                    self.bins.setdefault(vector_bin_ids[vector_id], []).append(found_ids[vector_id])
            # for vector_id in range(unique_vectors.shape[0]):
            #     if found_ids[vector_id] == -1: # not found
            # for bin_id, vector_ids in group_by_bins.items():
            #     bin_entries = self.bins.setdefault(bin_id, [])
            #     bin_new_tensor = unique_vectors[vector_ids]
            #     if len(bin_entries) == 0:
            #         found_ids = [-1] * bin_new_tensor.shape[0]
            #     else:
            #         found_ids = self.find_close(bin_entries, bin_new_tensor)
            #     missing_ids = get_missing_ids(found_ids)
            #     if len(missing_ids) > 0:
            #         new_ids = self.alloc_vectors(bin_new_tensor[missing_ids])
            #         found_ids = merge_ids(found_ids, new_ids)
            #         bin_entries.extend(new_ids)
            #     bin_found_ids.extend(found_ids)    
            #     del bin_new_tensor
            #     if len(bin_entries) >= self.max_bin_size and len(bin_entries) > len(self.bins.get(trigger_bin_id, [])):
            #         trigger_bin_id = bin_id
        trigger_bin_id = max(group_by_bins.keys(), key=lambda i: len(self.bins[i]))
        if len(self.bins.get(trigger_bin_id, [])) >= self.max_bin_size:
            self.rebuild(trigger_bin_id)
        return found_ids
    
    def query_points(self, query: torch.Tensor) -> list[int]:
        ''' O(1) '''
        bin_indices = self.get_bin_index(query)
        bin_ids = set([tuple(row) for row in bin_indices.tolist()])
        # for qid, bin_id in enumerate(bin_indices.tolist()):
        #       bin_ids.setdefault(tuple(bin_id), []).append(qid)
        del bin_indices
        bin_entries = [e for bin_index in bin_ids for e in self.bins.get(bin_index, [])]
        if len(bin_entries) > self.switch_to_all_cap * self.cur_id:
            bin_entries = None
        found_ids = self.find_close(bin_entries, query) # recompute stats if needed
        # qid_found_ids = {}
        # for bin_id, qids in bin_ids.items():
        #     present_ids = self.bins.get(bin_id, [])
        #     if len(present_ids) == 0:
        #         found_ids = [-1] * len(qids)
        #     else:
        #         found_ids = self.find_close(present_ids, query[qids])
        #     qid_found_ids.update(zip(qids, found_ids))
        # found_ids = [qid_found_ids[qid] for qid in range(query.shape[0])]
        return found_ids

    def query_range(self, qrange: torch.Tensor) -> list[int]:
        ''' Assumes that mapping to bin space is continuous: lines are left lines '''
        bin_qrange = self.get_bin_index(qrange)
        all_bin_ids_list = [bin_id for bin_id in self.bins.keys()]
        all_bin_ids = torch.tensor(all_bin_ids_list, dtype = torch.int64, device=qrange.device)
        bin_id_ids = find_in_range(all_bin_ids, bin_qrange)
        selected_bin_ids = [all_bin_ids_list[i] for i in bin_id_ids]
        # bin_ranges = product(*[list(range(min(b1, b2), max(b1, b2) + 1)) for b1, b2 in zip(min_bin, max_bin)])
        pass
        range_entries = [e for bin_index in selected_bin_ids for e in self.bins.get(bin_index, [])]
        del all_bin_ids, bin_qrange
        if len(range_entries) == 0:
            return []        
        found_ids = self.find_in_range(range_entries, qrange)
        return found_ids 

                        
class GridIndex(BinIndex):
    ''' Grid-based spatial index for approximate NN searc.
        Splits space onto bins of fixed size. Works only with points.
        Rebuild scales down the grid to satisfy max bin size in number of points.
    '''
    
    def __init__(self, epsilons: float | torch.Tensor = 1e-3, **kwargs):
        super().__init__(**kwargs)
        if not torch.is_tensor(epsilons):
            self.epsilons = torch.full((self.vectors.shape[-1], ), epsilons, dtype=self.vectors.dtype, device=self.vectors.device)
        else:
            self.epsilons = epsilons

    def get_bin_index(self, vectors: torch.Tensor) -> list[tuple]:
        ''' Get bin index for a given vector '''
        return torch.floor(vectors // self.epsilons).to(dtype=torch.int64)
    
    # def on_rebuild(self, trigger_bin_id: tuple):
    #     start_r_time = time.time()
    #     selection = self.get_vectors(self.bins[trigger_bin_id]) # get all vectors in the bin
    #     # balances = (selection - medians).sum(dim=0)
    #     # balances.abs_()
    #     # sort_ids = torch.argsort(balances)
    #     approx_num_dims = math.floor(math.log2(selection.shape[0] / self.max_bin_size)) + 1
    #     # approx_num_dims = max(1, approx_num_dims)
    #     selected_dims = torch.randint(selection.shape[-1], (approx_num_dims,), device=self.epsilons.device, dtype=torch.int64)
    #     medians = selection[:,selected_dims].median(dim = 0).values

    #     # selected_dims = sort_ids[:approx_num_dims] # take only first approx_num_dims dimensions
    #     bin_id_tensor = torch.tensor([trigger_bin_id[i] for i in selected_dims], dtype=self.epsilons.dtype, device=self.epsilons.device)
    #     bin_start = self.epsilons[selected_dims] * bin_id_tensor
    #     self.epsilons[selected_dims] = medians - bin_start
    #     # zero = torch.tensor(0, dtype=new_epsilons.dtype, device=new_epsilons.device)
    #     # self.epsilons = torch.where(torch.isclose(new_epsilons, zero, atol=self.atol, rtol=self.rtol),
    #     #                            self.epsilons, new_epsilons)
    #     end_r_time = time.time()    
    #     print("\tRebuild: ", end_r_time - start_r_time)
    #     pass

    def on_rebuild(self, trigger_bin_id: tuple):
        # start_r_time = time.time()
        selection = self.get_vectors(self.bins[trigger_bin_id]) # get all vectors in the bin
        medians = selection.median(dim = 0).values
        balances = (selection - medians).sum(dim=0)
        balances.abs_()
        sort_ids = torch.argsort(balances)
        approx_num_dims = math.floor(math.log2(selection.shape[0] / self.max_bin_size)) + 1
        selected_dims = sort_ids[:approx_num_dims] # take only first approx_num_dims dimensions
        bin_id_tensor = torch.tensor([trigger_bin_id[i] for i in selected_dims], dtype=self.epsilons.dtype, device=self.epsilons.device)
        bin_start = self.epsilons[selected_dims] * bin_id_tensor
        self.epsilons[selected_dims] = medians[selected_dims] - bin_start
        # zero = torch.tensor(0, dtype=new_epsilons.dtype, device=new_epsilons.device)
        # self.epsilons = torch.where(torch.isclose(new_epsilons, zero, atol=self.atol, rtol=self.rtol),
        #                            self.epsilons, new_epsilons)
        # end_r_time = time.time()    
        # print("\tRebuild: ", end_r_time - start_r_time)        
        pass    
        
class MBR:        

    def __init__(self, mbr: torch.Tensor, points: Optional[torch.Tensor] = None):
        ''' mbr shape (2, *k, dims), points shape (n, *k, dims) '''
        self.mbr = mbr
        self.points = points
        self.point_ids: Optional[list[int]] = None
        self._area: Optional[float] = None 
            
    def has_points(self) -> bool:
        return self.points is not None

    def get_min(self) -> torch.Tensor:
        return self.mbr[0]
    
    def get_max(self) -> torch.Tensor:
        return self.mbr[1]
        
    def area(self) -> float:        
        if self._area is None:
            self._area = torch.prod(self.get_max() - self.get_min()).item()
        return self._area
    
    def enlarge(self, *p: "MBR") -> "MBR":
        if len(p) == 0:
            return self
        mbrs = torch.stack([self.mbr, *(x.mbr for x in p)])
        new_mbr = torch.empty((2, *mbrs.shape[1:]), dtype=mbrs.dtype, device=mbrs.device)
        new_mbr[0] = mbrs[:, 0].min(dim=0).values
        new_mbr[1] = mbrs[:, 1].max(dim=0).values
        new_mbr = MBR(mbrs) # do not pass points here 
        return new_mbr
    
    def enlargement(self, p: "MBR") -> tuple[float, float, "MBR"]:
        ''' Resize MBR to cover the point.
        '''
        new_mbr = self.enlarge(p)
        new_area = new_mbr.area()
        old_area = self.area()
        return new_area - old_area, old_area, new_mbr     
    
    def intersects(self, other: "MBR") -> bool:
        ''' Check if this MBR intersects with another MBR. '''
        return torch.all(self.get_min() <= other.get_max()) and torch.all(other.get_min() <= self.get_max())

@dataclass(eq=False, unsafe_hash=False)
class RTreeNode:
    mbr: MBR 
    children: list[Union["RTreeNode", MBR]]

    def is_leaf(self) -> bool:
        return isinstance(self.children[0], MBR) 

    def create_node_from_child(self, i: int) -> "RTreeNode":
        ''' Create node from child at index i. '''
        if self.is_leaf():
            return RTreeNode(self.children[i], [self.children[i]])
        return RTreeNode(self.children[i].mbr, [self.children[i]])
        
def linear_split(node: RTreeNode, min_children: int = 1) -> list[RTreeNode]:
    get_mbr = (lambda x: x) if node.is_leaf() else (lambda x: x.mbr)
    min_tensors = torch.stack([get_mbr(x).get_min() for x in node.children])
    max_tensors = torch.stack([get_mbr(x).get_max() for x in node.children])
    Li = min_tensors.argmin(dim=0)
    Hi = max_tensors.argmax(dim=0)
    # separations = (max_tensors[H] - min_tensors[L]) / (node.mbr.max_point - node.mbr.min_point)
    separations = (max_tensors[Hi] - min_tensors[Li]) / (node.mbr.get_max() - node.mbr.get_min())
    mean_separations = separations.mean(dim=0)
    max_sep_dim_id = mean_separations.argmax().item()
    selected_l_id = Li[max_sep_dim_id].item()
    selected_h_id = Hi[max_sep_dim_id].item()
    if selected_l_id == selected_h_id:
        pass
    child1 = node.create_node_from_child(selected_l_id)
    child2 = node.create_node_from_child(selected_h_id)
    new_children = [child1, child2]
    left_children = [c for i, c in enumerate(node.children) if i != selected_l_id and i != selected_h_id]
    for i, child in enumerate(left_children):
        left_count = len(left_children) - i - 1
        min_child = next((c for c in new_children if len(c.children) + left_count <= min_children), None)
        if min_child is not None: # all rest should go to this child
            left_children_to_append = left_children[i:]
            if node.is_leaf():
                left_children_mbrs = left_children_to_append
            else:
                left_children_mbrs = (c.mbr for c in left_children_to_append)
            min_child.children.extend(left_children_to_append)
            min_child.mbr = min_child.mbr.enlarge(*left_children_mbrs)
            break
        else:
            _, _, new_mbr, selected_child = min(((c_enl, len(c.children), new_mbr, c) for c in new_children for c_enl, _, new_mbr in [c.mbr.enlargement(child)]), key = lambda x: (x[0], x[1]))
            selected_child.children.append(child)
            selected_child.mbr = new_mbr
    return new_children

class RTreeIndex(SpatialIndex):
    ''' R-Tree spatial index for NN search.
        It is a tree structure where each node contains a minimum bounding rectangle (MBR) that covers its children.
        The MBR is defined by the minimum and maximum coordinates in each dimension.
    '''
    def __init__(self, min_children: int = 2, max_children: int = 10, split_strategy = linear_split, **kwargs):
        super().__init__(**kwargs)
        self.min_children = min_children
        self.max_children = max_children
        self.split_strategy = split_strategy
        self.root: RTreeNode | None = None

    def _insert(self, node: RTreeNode, mbr: MBR) -> tuple[list[RTreeNode], list[int]]:
        ''' Insert point into the R-Tree node. '''
        if node.is_leaf():
            point_ids = [point_id for c in node.children for point_id in (c.point_ids or [])]
            if len(point_ids) > 0:
                found_ids = self.find_close(point_ids, mbr.points)
            else:
                found_ids = [-1] * mbr.points.shape[0]
            missing_ids = get_missing_ids(found_ids)
            if len(missing_ids) > 0:
                new_ids = self.alloc_vectors(mbr.points[missing_ids])
                found_ids = merge_ids(found_ids, new_ids)
            mbr.point_ids = found_ids
            del mbr.points
            mbr.points = None
            node.children.append(mbr)
            replacement = []
            if len(node.children) > self.max_children:
                replacement = self.split_strategy(node, self.min_children)
            return replacement, mbr.point_ids
        else:
            _, _, new_mbr, child_i = min(((c_enl, c_ar, new_mbr, i) for i, c in enumerate(node.children) for c_enl, c_ar, new_mbr in [c.enlargement(mbr)]), key = lambda x: (x[0], x[1]))
            selected_child = node.children[child_i]
            selected_child.mbr = new_mbr
            replacement, point_ids = self._insert(selected_child, mbr)
            if len(replacement) > 0: # overflow propagation
                node.children = [*node.children[:child_i], *replacement, *node.children[child_i+1:]]
                if len(node.children) > self.max_children:
                    replacement = self.split_strategy(node, self.min_children)
                else:
                    replacement = []
            return replacement, point_ids

    def _insert_distinct(self, unique_vectors: torch.Tensor) -> list[int]:
        ''' Inserts one point (rects are not supported yet) ''' 
        mbr = MBR(get_mbr(unique_vectors), unique_vectors)
        if self.root is None:
            new_ids = self.alloc_vectors(unique_vectors)
            mbr.point_ids = new_ids
            self.root = RTreeNode(mbr, [mbr])
            return new_ids
        else:
            replacement, vector_id = self._insert(self.root, mbr) 
            if len(replacement) > 0: # root split - need to create new root
                self.root = RTreeNode(self.root.mbr, replacement)             
            return vector_id  

    def _query(self, node: RTreeNode, mbr: MBR) -> Generator[int, None, None]:
        if node.is_leaf():
            yield from (c.point_ids for c in node.children if c.intersects(mbr))
        else:
            for c in node.children:
                if c.mbr.intersects(mbr):
                    yield from self._query(c, mbr)
    
    def query_points(self, query: torch.Tensor) -> list[int]:
        mbr = MBR(get_mbr(query))
        found_point_ids = list(self._query(self.root, mbr))
        point_tensor = self.get_vectors(found_point_ids)
        found_ids = self.find_close(point_tensor, query)
        return found_ids
    
    def query_range(self, qrange: torch.Tensor) -> list[int]:
        mbr = MBR(qrange)
        found_ids = list(self._query(self.root, mbr))
        return found_ids

def pack_bits(bits: torch.Tensor, dtype = torch.int64) -> torch.Tensor:
    ''' Splits bits (*ns, m) 0 1 uint8 tensor into tensor of shape (*ns, ceil(m / size(dtype))) '''
    sz = torch.iinfo(dtype).bits
    bitmask = (1 << torch.arange(sz - 1, -1, -1, dtype = dtype, device = bits.device)) # shape [64]
    max_packed_values = math.ceil(bits.shape[-1] / sz) # number of packed values
    packed = []
    # bits_tp = bits.to(dtype=tp, device=bits.device) # convert bits to target type
    for i in range(max_packed_values):
        start = i * sz
        end = start + sz
        if end > bits.shape[-1]:
            end = bits.shape[-1]
        packed_chunk = torch.sum(bits[..., start:end] * bitmask[:end - start], dtype = dtype, dim=-1)
        packed.append(packed_chunk)
    return torch.stack(packed, dim=-1)


def unpack_bits(packed: torch.Tensor, clamp_sz: int) -> torch.Tensor:
    ''' Converts packed (list of shape (n) tensors int64) back to (n, sz) shape uint8 of 0 1 values'''
    sz = torch.iinfo(packed.dtype).bits
    bitmask = (1 << torch.arange(sz - 1, -1, -1, device=packed.device, dtype=packed.dtype))  # Shape: (num_bits,)
    unpacked = ((packed.unsqueeze(-1) & bitmask) != 0).view(*packed.shape[:-1], -1)  # Shape: (N, num_bits), bool tensor
    return unpacked[..., :clamp_sz]  # Convert to uint8 (0/1 values)

# t1 = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0], dtype=torch.uint8)
# t1 = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.uint8)
# t1 = torch.tensor([True, False, True, False], dtype=torch.bool)
# res = pack_bits(t1, dtype=torch.int8)
# t2 = unpack_bits(res, clamp_sz=t1.shape[-1])
# assert torch.equal(t1, t2), "Packing and unpacking failed, tensors are not equal."
# pass
        
class InteractionIndex(BinIndex):
    ''' Maps semantics to binary vector based on dynamically computed epsilons and given target. 
        One dim is one test and 0 means we far from passing the test, 1 - close.
        leaf (one interaction vector bin) splits when it has many semantics
    '''
    def __init__(self, target: torch.Tensor, pack_dtype = torch.uint64, **kwargs):
        super().__init__(**kwargs)
        self.epsilons: int | torch.Tensor = 0
        sz = torch.iinfo(pack_dtype).bits
        self.int_dims = math.ceil(self.vectors.shape[-1] / sz)
        # self.interactions = torch.zeros((self.capacity, int_dims), dtype=pack_dtype, device=self.vectors.device)
        self.target = target
        self.pack_dtype = pack_dtype
        # self.iid_to_vids: dict[int, list[int]] = {} # interaction id to vector ids
        # self.vid_to_iid: dict[int, int] = {} # vector id to interaction id
    
    def get_bin_index(self, vectors: torch.Tensor) -> list[tuple]:
        distances = torch.abs(vectors - self.target)
        interactions = (distances <= self.epsilons)
        ints = pack_bits(interactions, dtype = self.pack_dtype)
        del interactions, distances
        return ints    
    
    def on_rebuild(self, trigger_bin_id: tuple):
        vectors = self.get_vectors(self.bins[trigger_bin_id])
        distances = torch.abs(vectors - self.target)
        new_epsilons = distances.mean(dim=0)
        zero = torch.tensor(0, dtype=new_epsilons.dtype, device=new_epsilons.device)
        self.epsilons = torch.where(torch.isclose(new_epsilons, zero, atol=self.atol, rtol=self.rtol), self.epsilons, new_epsilons)
        del vectors, distances

    # def get_interaction(self, vector_id: int) -> torch.Tensor:
    # TODO: if necessary

    
class RCosIndex(BinIndex):
    ''' Represents torch.Tensor with only radius vector and cosine distance to target vector 
        Splits spalce onto cones by angle and radius.
    ''' 
    def __init__(self, target: torch.Tensor, epsilons: float | torch.Tensor = 1e-1, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.target_norm = torch.norm(self.target)
        assert not torch.isclose(self.target_norm, 0, atol=self.atol, rtol=self.rtol), "Target vector cannot be zero vector."
        if torch.is_tensor(epsilons):
            self.epsilons = epsilons
        else:
            self.epsilons = torch.full(2, epsilons, dtype=self.vectors.dtype, device=self.vectors.device)

    def get_bin_index(self, vectors: torch.Tensor) -> list[tuple]:
        ''' Get bin index for a given vector '''
        norms = torch.norm(vectors - self.target, dim=-1)
        zero = torch.tensor(0, dtype=norms.dtype, device=norms.device)
        cos_distance = 1 - torch.where(torch.isclose(norms, zero, atol = self.atol, rtol=self.rtol)
                                       , 1, torch.dot(vectors, self.target) / (norms *self.target_norm))
        norm_bin_index = torch.floor(norms / self.epsilons[0]).to(dtype=torch.int64)
        cos_bin_index = torch.floor(cos_distance / self.epsilons[1]).to(dtype=torch.int64)
        bin_index = torch.stack((norm_bin_index, cos_bin_index), dim=-1) # shape (n, 2)
        del norms, cos_distance
        return bin_index
    
    def on_rebuild(self, trigger_bin_id: tuple):
        bin_tensor = self.get_vectors(self.bins[trigger_bin_id])
        norms = torch.norm(bin_tensor - self.target, dim=1)
        zero = torch.tensor(0, dtype=norms.dtype, device=norms.device)
        cos_distance = 1 - torch.where(torch.isclose(norms, zero, atol = self.atol, rtol=self.rtol)
                                       , 1, torch.dot(bin_tensor, self.target) / (norms *self.target_norm))        
        new_norm_epsilon = (norms.max() - norms.min()) / 2
        new_cos_epsilon = (cos_distance.max() - cos_distance.min()) / 2
        new_epsilon = torch.tensor([new_norm_epsilon, new_cos_epsilon], dtype=self.vectors.dtype, device=self.vectors.device)
        self.epsilons[:] = torch.where(torch.isclose(new_epsilon, 0, atol=self.atol, rtol=self.rtol), self.epsilons[0], new_epsilon)
        
class SpearmanCorIndex(BinIndex):
    ''' vector to spearman correlation with target ''' 

    def __init__(self, target: torch.Tensor, epsilon = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.target = target

    def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_rank = torch.argsort(torch.argsort(x))
        y_rank = torch.argsort(torch.argsort(y))

        # Compute Pearson correlation on the ranks
        x_mean = x_rank.float().mean()
        y_mean = y_rank.float().mean()
        numerator = torch.sum((x_rank - x_mean) * (y_rank - y_mean))
        denominator = torch.sqrt(torch.sum((x_rank - x_mean)**2) * torch.sum((y_rank - y_mean)**2))

        return numerator / denominator
    
    def get_bin_index(self, vectors: torch.Tensor) -> list[tuple]:
        cors: torch.Tensor = self.spearman_correlation(vectors, self.target)
        cors.abs_()
        bin_id = torch.floor(cors / self.epsilon).to(dtype=torch.int64)
        return bin_id
    
    def on_rebuild(self, trigger_bin_id: tuple):
        bin_tensor = self.get_vectors(self.bins[trigger_bin_id])
        cors: torch.Tensor = self.spearman_correlation(bin_tensor, self.target)
        cors.abs_()
        new_epsilon = (cors.max() - cors.min()) / 2
        if torch.isclose(new_epsilon, 0, atol=self.atol, rtol=self.rtol):
            new_epsilon = self.epsilon / 2 
        else:
            self.epsilon = new_epsilon
        pass

def test_storage(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cpu"):
    storage = VectorStorage(capacity, dims, dtype=dtype, device=device)
    all_ids = []
    all_chunks = []
    for chunk_sz in range(1, 10):
        chunk = torch.randn((chunk_sz, dims), dtype=dtype, device=device)
        ids = storage.alloc_vectors(chunk)
        all_ids.append(ids)
        all_chunks.append(chunk)
    assert storage.cur_id == sum(len(l) for l in all_ids), "Storage did not allocate all vectors correctly."
    assert list(range(storage.cur_id)) == [el for l in all_ids for el in l], "Storage allocated vectors with wrong ids."
    chunk_tensor = torch.cat(all_chunks, dim=0)
    assert torch.equal(storage.get_vectors(None), chunk_tensor), "Storage did not return all vectors"
    selected_ids = [el for l in all_ids[1:-1] for el in l]
    selected_vectors = storage.get_vectors(selected_ids)
    chunk_tensor2 = torch.cat(all_chunks[1:-1], dim=0)
    assert torch.equal(selected_vectors, chunk_tensor2), "Storage did not return correct vectors for selected ids."
    chunk_id = 5
    assert torch.equal(storage.get_vectors((all_ids[chunk_id][0], all_ids[chunk_id][-1] + 1)), all_chunks[chunk_id]), "Storage did not return correct vector for id 0."
    el_ids = [1, 5, 42]
    query = storage.get_vectors(el_ids)
    close_ids = storage.find_close(None, query)
    for close_id, el_id in zip(close_ids, el_ids):
        if close_id != el_id:
            close_vector = storage.get_vectors(close_id)
            el_vector = storage.get_vectors(el_id)
            assert torch.allclose(close_vector, el_vector, rtol=storage.rtol, atol=storage.atol), "Storage did not return close vectors correctly."
    mbr = storage.find_mbr(None)
    id_range = (13, 31)
    in_range = storage.find_in_range(id_range, mbr)
    # all_vectors = storage.get_vectors(None)
    assert in_range == list(range(*id_range)), "Storage did not return all"

    pass

import time
def test_spatial_index(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=4096, query_batch_size=256, dim_batch_size=4,
                        num_points_per_group = 100, num_groups = 100,
                        min_r = 0.3, max_r = 0.7, time_deltas = False,
                        index = SpatialIndex):
    
    idx = index(capacity=capacity, dims=dims, dtype=dtype, device=device,
                        store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                        dim_batch_size = dim_batch_size)
    step = 1 / num_groups
    all_ids_grouped = []
    all_ids = []
    all_points = []
    start_time = time.time()
    times = []
    for i in range(num_groups):
        pi = torch.rand((num_points_per_group, dims), dtype=dtype, device=device)
        new_ids = idx.insert(pi)
        all_ids.extend(new_ids)
        all_ids_grouped.append(new_ids)
        all_points.extend(pi.tolist())
        iter_end = time.time()
        total_duration = iter_end - start_time
        if time_deltas:
            start_time = iter_end
        times.append(total_duration)
        print(f"Inserted group {i + 1:02}/{num_groups} in {total_duration:06.2f} seconds, "
              f"total {idx.cur_id:06} points, {store_batch_size:04}:{query_batch_size:03}:{dim_batch_size:03}")
    search_points = torch.randint(idx.cur_id, (100,)).tolist()
    search_tensors = idx.get_vectors(search_points)
    found_ids = idx.find_close(None, search_tensors)
    # should_be_ids = [all_ids[sp] for sp in search_points]
    assert sorted(search_points) == sorted(found_ids), "Spatial index did not return correct ids for search points."
    # range_query = torch.full((2, dims), 0, dtype=dtype, device=device)
    # range_query[0] = min_r
    # range_query[1] = max_r
    # found_ids = idx.find_in_range(None, range_query)
    # expected_ids = [el for i in range(int(min_r / step), int(max_r / step) + 1) for el in all_ids_grouped[i]]
    # assert sorted(found_ids) == sorted(expected_ids), "Spatial index did not return correct ids in range"

    return times
    # assert idx.cur_id == num_points_per_group * num_groups, "Spatial index did not allocate all vectors correctly."
    
def test_spatial_index_query(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=4096, query_batch_size=256, dim_batch_size=4,
                        num_points_per_group = 1, num_groups = 100,
                        time_deltas = False, index = SpatialIndex):
    
    idx = index(capacity=capacity, dims=dims, dtype=dtype, device=device,
                        store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                        dim_batch_size = dim_batch_size)
    start_insert_time = time.time()
    idx.insert(torch.rand((capacity, dims), dtype = dtype, device=device))
    # for bin_id, point_ids in idx.bins.items():
    #     point_tensor = idx.get_vectors(point_ids)
    #     bin_tensor = torch.tensor(bin_id, dtype=idx.epsilons.dtype, device=idx.epsilons.device)
    #     bin_min = bin_tensor * idx.epsilons
    #     bin_max = bin_min + idx.epsilons
    #     assert torch.all((point_tensor >= bin_min) & (point_tensor <= bin_max)), "Grid index did not insert points in correct bins."    
    #     pass
    start_time = time.time()
    print(f"Inserted {idx.cur_id:06} points in {start_time - start_insert_time:06.2f} seconds")
    times = []
    for i in range(num_groups):
        # ids_range = (0, (i + 1) * num_points_per_group)
        orig_ids = torch.randint(idx.cur_id, ((i + 1) * num_points_per_group,)).tolist()
        orig_ids.sort()
        query_points = idx.get_vectors(orig_ids)
        found_ids = idx.query_points(query_points)
        # found_ids = []
        # for query_point in query_points:
        #     one_found_ids = idx.query_points(query_point.unsqueeze(0)) # query one point at a time
        #     found_ids.extend(one_found_ids)
        found_ids.sort()
        assert found_ids == orig_ids, f"Spatial index did not return correct ids for group"
        iter_end = time.time()
        total_duration = iter_end - start_time
        if time_deltas:
            start_time = iter_end
        times.append(total_duration)
        print(f"Queried group {i + 1:02}/{num_groups} in {total_duration:06.2f} seconds, "
              f"total {len(orig_ids):06} points, {store_batch_size:04}:{query_batch_size:03}:{dim_batch_size:03}")

    return times
    # assert idx.cur_id == num_points_per_group * num_groups, "Spatial index did not allocate all vectors correctly."

def test_time(f, **arg_combs):    
    plt.figure(figsize=(10, 6))
    keys = list(arg_combs.keys())
    for arg_comb in product(*arg_combs.values()):
        times = f(**{k: v for k, v in zip(keys, arg_comb)})
        plt.plot(times, label=f"({arg_comb})")
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

def test_grid_index(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8,
                        num_points_per_group = 100, num_groups = 100,
                        min_r = 0.3, max_r = 0.7, time_deltas = False):
    idx = GridIndex(epsilons=1, max_bin_size = 2000, capacity=capacity, dims=dims, dtype=dtype, device=device,
                    store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                    dim_batch_size = dim_batch_size)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids1 = idx.insert(p)
    p_ids2 = idx.insert(p)
    assert sorted(p_ids1) == sorted(p_ids2), "Grid index did not return same ids for same points."
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids3 = idx.insert(p) # should trigger rebuild
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids4 = idx.insert(p)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids5 = idx.insert(p)

    v = idx.get_vectors(p_ids1)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids1), "Grid index did not return correct"
    v = idx.get_vectors(p_ids2)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids2), "Grid index did not return correct"
    v = idx.get_vectors(p_ids3)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids3), "Grid index did not return correct"
    v = idx.get_vectors(p_ids4)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids4), "Grid index did not return correct"
    
    r_ids = idx.query_range(torch.tensor([[0] * dims, [1] * dims], dtype=dtype, device=device))
    assert sorted(r_ids) == list(range(idx.cur_id)), "Grid index did not return all ids in range query."

    
    pass 

# visualize_2d_frame_id = 0

def visualize_2d(x, y, rects=None, epsilons=None, xrange = None, yrange = None):
    """
    Visualizes scattered points, rectangles, and an optional grid.

    Args:
        x (list[float]): x-coordinates of scattered points.
        y (list[float]): y-coordinates of scattered points.
        rects (list[tuple[float, float, float, float]]): List of rectangles, each defined as (x_min, y_min, width, height).
        epsilons (tuple[float, float]): Grid spacing for x and y axes (optional).
    """
    # global visualize_2d_frame_id
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.ticker as ticker
    plt.ion()
    plt.clf()
    # plt.figure(figsize=(10, 6))
    
    # Plot scattered points
    plt.scatter(x, y, color="black", s = 3)
    
    # Plot grid
    if epsilons:
        x_epsilon, y_epsilon = epsilons
        plt.gca().set_xticks(np.arange(math.floor(xrange[0] / x_epsilon), math.floor(xrange[1] / x_epsilon) + 1) * x_epsilon)
        plt.gca().set_yticks(np.arange(math.floor(yrange[0] / y_epsilon), math.floor(yrange[1] / y_epsilon) + 1) * y_epsilon)
        plt.gca().set_xticklabels([])
        plt.gca().set_xticklabels([])
        plt.gca().tick_params(labelbottom=False, labelleft=False)
        plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    
    # Plot rectangles
    if rects:
        for rect in rects:
            x_min, y_min, width, height = rect
            rect_patch = patches.Rectangle((x_min, y_min), width, height, 
                                           linewidth=1, edgecolor="red", facecolor="red", alpha=0.2)
            plt.gca().add_patch(rect_patch)
    
    # Add labels and legend
    # plt.xlabel("X")
    # plt.ylabel("Y")
    if xrange is not None:
        plt.xlim(xrange)
    if yrange is not None:
        plt.ylim(yrange)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.02f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.02f'))
    # plt.title("2D Visualization of Points and Rectangles")
    # plt.legend()
    plt.tight_layout()
    # plt.ioff()
    # plt.show()
    # if visualize_2d_frame_id == 0:
    #     plt.get_current_fig_manager().full_screen_toggle()
    # visualize_2d_frame_id += 1
    plt.pause(3)

# Example usage
# x = [1, 2, 3, 4, 5]
# y = [5, 4, 3, 2, 1]
# rects = [(1.5, 1.5, 2, 2), (3.5, 3.5, 1, 1)]
# epsilons = (1, 1)

# visualize_2d(x, y, rects=rects, epsilons=epsilons)
# pass

def viz_grid(idx: GridIndex):
    epsilonx, epsilony = idx.epsilons.tolist()
    vectors = idx.get_vectors(None)
    rects = [(bin_id_x * epsilonx, bin_id_y * epsilony, epsilonx, epsilony) for bin_id_x, bin_id_y in idx.bins.keys()]
    xs = vectors[:, 0].tolist()
    ys = vectors[:, 1].tolist()
    visualize_2d(xs, ys,
                 rects=rects,
                 epsilons=(epsilonx, epsilony),
                 xrange=(0, 1), yrange=(0, 1))
    
def test_grid_distr(capacity = 100_000, dims = 2, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8,
                        num_groups = 10):
    idx = GridIndex(epsilons=1, max_bin_size = 10, capacity=capacity, dims=dims, dtype=dtype, device=device,
                    store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                    dim_batch_size = dim_batch_size)
    for num_groups in range(num_groups):
        means = torch.rand(dims, dtype=dtype, device=device)
        stds = torch.full((100, dims), 0.05, dtype=dtype, device=device)
        distr = torch.normal(mean = means, std = stds)
        distr.clamp_(0, 1)
        # distr = shifts + torch.rand((100, dims), dtype=dtype, device=device) * 0.1
        idx.insert(distr)
        viz_grid(idx)    
    # plt.pause(5)
    # plt.show()

    pass

if __name__ == "__main__":
    # test_storage()    
    # test_grid_index()
    test_grid_distr()
    # test_time(partial(test_spatial_index_query,
    #                     # index = SpatialIndex,
    #                     # index = partial(GridIndex, epsilons=1, max_bin_size = 64,
    #                     #                 switch_to_all_cap = 0.9),
    #                     index = partial(RTreeIndex, min_children=64, max_children=128, split_strategy=linear_split)
    #                   ), 
    #             store_batch_size = [2048, ], #2048, 4096], 
    #             dim_batch_size = [4, ], # 8, 16],
    #             )
    # test_time(partial(test_spatial_index,
    #                     # index = SpatialIndex,
    #                     index = partial(GridIndex, epsilons=1, max_bin_size = 64),
    #                   ), 
    #             store_batch_size = [1024, ], #2048, 4096], 
    #             dim_batch_size = [4, ], # 8, 16],
    #             )
    # plt.ioff()
    pass