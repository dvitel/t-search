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
from typing import Callable, Generator, Literal, Optional
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
        return vectors[ids] # return a view by index 
    if len(ids) == 1:
        return vectors[ids[0]].unsqueeze(0) # also view
    return vectors[ids] # this will new tensor with values copied from mindices        

def find_pred(store: torch.Tensor, query: torch.Tensor, 
                pred: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                store_batch_size = 1024, query_batch_size = 256, dim_batch_size = 8,
                return_shape: Literal["mask", "ids"] = "ids") -> torch.Tensor:
    ''' store shape (n, *k, dims), query shape (m, *k, dims), 
        computes mask (n, m) of True/False - x 'row' is close to y 'row' 
        returns tensor of row ids of store where query matches or -1 if no match or mask
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
                     query_start:query_end] &= mask_per_el.all(dim = tuple(range(2, mask_per_el.ndim))) # (n, m)
                del mask_per_el
                if not torch.any(mask[store_start:store_end, query_start:query_end]):
                    break
    if return_shape == "mask":
        return mask
    store_ids, query_ids = torch.where(mask) # tuple row (ids1, ids2) where idsX has number of matches and indexes for dimension X
    ids = torch.full((mask.shape[-1],), -1, dtype = torch.int64, device=mask.device)
    ids[query_ids] = store_ids
    # del mask
    # ids_list = ids.tolist()
    # del ids 
    return ids

def find_close(store: torch.Tensor, query: torch.Tensor, rtol=1e-5, atol=1e-4,
                    store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 64,
                    return_shape: Literal["mask", "ids"] = "ids") -> torch.Tensor:
    ''' store shape (n, *k, dims), query shape (m, *k, dims), 
        computes mask (n, m) of True/False - x 'row' is close to y 'row' 
        returns list of row ids of store where query matches or -1 if no match
    '''
    found_ids = find_pred(store, query,
                          lambda s, q: torch.isclose(s, q, rtol=rtol, atol=atol), 
                          store_batch_size=store_batch_size, query_batch_size=query_batch_size, 
                          dim_batch_size=dim_batch_size, return_shape = return_shape)
    return found_ids

def find_closest(store: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    ''' Store (n, dims), query (m, dims). For each query m, searches closest point in store '''
    store_sq = (store ** 2).sum(dim=1, keepdim=True) # (n, 1)
    query_sq = (query ** 2).sum(dim=1, keepdim=True) # (m, 1)
    prod = torch.mm(store, query.t()) # (n, m)
    squared_dists = store_sq - 2 * prod + query_sq.t() # (n, m)
    found_ids = torch.argmin(squared_dists, dim=0) # (m,)
    distinct_ids = torch.unique(found_ids)

    return distinct_ids

def find_equal(store: torch.Tensor, query: torch.Tensor,
               store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 64,
               return_shape: Literal["mask", "ids"] = "ids") -> torch.Tensor:
    found_ids = find_pred(store, query, 
                          lambda s, q: s == q,
                          store_batch_size=store_batch_size, query_batch_size=query_batch_size, dim_batch_size=dim_batch_size,
                          return_shape = return_shape)
    return found_ids

def find_in_ranges(store: torch.Tensor, query: torch.Tensor,
               store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 64) -> torch.Tensor:
    ''' store (n, *l, dims), query (2, k, *l, dims) '''
    query_t = query.transpose(0, 1)  # (k, 2, *l, dims)
    range_mask = find_pred(store, query_t, # s (n0, 1, dims), q, (1, k0, 2, dims)
                          lambda s, q: (q[:, :, 0] <= s) & (s <= q[:, :, 1]),
                          store_batch_size=store_batch_size, query_batch_size=query_batch_size, dim_batch_size=dim_batch_size,
                          return_shape="mask")
    return range_mask

def find_intersects(store: torch.Tensor, query: torch.Tensor,
               store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 64) -> torch.Tensor:
    ''' store (2, n, *l, dims), query (2, k, *l, dims) '''
    store_t = store.transpose(0, 1)  # (2, n, *l, dims)
    query_t = query.transpose(0, 1)  # (2, k, *l, dims)
    range_mask = find_pred(store_t, query_t, # s (n0, 1, 2, dims), q, (1, k0, 2, dims)
                          lambda s, q: (q[:, :, 0] <= s[:, :, 1]) & (s[:, :, 0] <= q[:, :, 1]),
                          store_batch_size=store_batch_size, query_batch_size=query_batch_size, dim_batch_size=dim_batch_size,
                          return_shape="mask")
    return range_mask

def find_in_range(store: torch.Tensor, query: torch.Tensor,
                    store_batch_size = 1024, dim_batch_size = 64) -> torch.Tensor:
    ''' query shape is (2, *x.shape[1:]) - one per min and max 
        m - number of ranges to test. Usually m = 1, then 
    '''
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
    return ids

def get_mbr(vectors:torch.Tensor) -> torch.Tensor:
    if len(vectors.shape) == 1:  # single vector
        return torch.stack((vectors, vectors))
    mbr = torch.empty((2, vectors.shape[1]), dtype=vectors.dtype, device=vectors.device)
    mbr[0] = vectors.min(dim=0).values
    mbr[1] = vectors.max(dim=0).values    
    return mbr

def remap_ids(local_ids: torch.Tensor, ids: None | tuple[int, int] | list[int], ) -> list[int]:
    if ids is None:
        return local_ids.tolist()
    found_mask = local_ids >= 0
    if isinstance(ids, tuple):
        local_ids[found_mask] += ids[0] 
        return local_ids.tolist()
    # here ids is list - should we resort to tensor? for later 
    ids_tensor = torch.tensor(ids, dtype=torch.int64, device=local_ids.device)
    local_ids_masked = local_ids[found_mask]
    local_ids[found_mask] = ids_tensor[local_ids_masked]
    del ids_tensor, local_ids_masked
    return local_ids.tolist()

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
                 store_batch_size: int = 1024, query_batch_size: int = 128, dim_batch_size: int = 8, **kwargs):
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
    
    def _alloc_vectors(self, vectors: torch.Tensor) -> list[int]:
        ''' Adds vectors (n, dims) to storage and returns new ids. '''
        cur_end = self.cur_id + vectors.shape[0]
        if cur_end > self.capacity: # reallocate memory 
            new_vectors = torch.empty((self.capacity * 2, vectors.shape[1]), dtype=vectors.dtype, device=vectors.device)
            new_vectors[:self.cur_id] = self.vectors[:self.cur_id]
            old_vectors = self.vectors
            self.vectors = new_vectors
            del old_vectors
        self.vectors[self.cur_id:cur_end] = vectors
        vector_ids = list(range(self.cur_id, self.cur_id + vectors.shape[0]))
        self.cur_id += vectors.shape[0]
        # if self.cur_id - self.stats.num_vectors >= self.stats_batch_size:
        #     self.stats.recompute(dim_delta = self.dim_delta)
        return vector_ids
    
    def query_points(self, points: torch.Tensor, 
                     atol: float | None = None, rtol: float | None = None) -> list[int]:
        ''' O(n). Return id of vectors in index if present. Empty tensor otherwise. '''
        return self.find_close(None, points, atol=atol, rtol=rtol) # q here is one point among N points of all_vectors of shape (N, ..., dims)
    
    def query_range(self, qrange: torch.Tensor) -> list[int]:
        ''' O(n). Returns ids stored in the index, shape (N), N >= 0 is most cases.
            qrange[0] - mins, qrange[1] - maxs, both of shape (dims).
        '''
        return self.find_in_range(None, qrange)

    # def query_closest(self, points: torch.Tensor, obj_ids: torch.Tensor | None = None) -> list[int]:
    #     return self.find_closest(None, points, obj_ids)
        
    def insert(self, vectors: torch.Tensor) -> list[int]:
        return self._alloc_vectors(vectors)
    
    # def _find(self, ids: None | tuple[int, int] | list[int], op, *args) -> list[int]:
        
    #     selection = self.get_vectors(ids)        
    #     permute_dim_id = lambda x:x*self.dim_delta 
    #     if self.stats and self.stats.var_dim_permutation:
    #         permute_dim_id = lambda x: self.stats.var_dim_permutation[x]
    #     found_ids = op(selection, *args, ids, dim_delta=self.dim_delta, 
    #                         permute_dim_id = permute_dim_id)
    #     return found_ids
    
    def find_close(self, ids: None | tuple[int, int] | list[int], q: torch.Tensor,
                    atol: float | None = None, rtol: float | None = None) -> list[int]:
        selection = self.get_vectors(ids)
        found_ids = find_close(selection, q, rtol=(rtol or self.rtol), atol=(atol or self.atol),
                               store_batch_size=self.store_batch_size, 
                               query_batch_size=self.query_batch_size, 
                               dim_batch_size=self.dim_batch_size)
        del selection
        return remap_ids(found_ids, ids)

    def find_closest(self, ids: None | tuple[int, int] | list[int], q: torch.Tensor,
                        obj_ids: torch.Tensor | None = None) -> list[int]:
        selection = self.get_vectors(ids)
        if obj_ids is None:
            store = selection
            query = q
        else:
            store = selection[:, obj_ids]
            query = q[:, obj_ids]
        found_ids = find_closest(store, query)
        del selection
        if obj_ids is not None:
            del store, query
        return remap_ids(found_ids, ids)
    
    def find_in_range(self, ids: None | tuple[int, int] | list[int], query: torch.Tensor) -> list[int]:
        selection = self.get_vectors(ids)
        found_ids = find_in_range(selection, query, store_batch_size=self.store_batch_size,
                                  dim_batch_size=self.dim_batch_size)
        del selection
        return remap_ids(found_ids, ids)

    def find_in_ranges(self, ids: None | tuple[int, int] | list[int], q: torch.Tensor) -> list[int]:
        selection = self.get_vectors(ids)
        mask = find_in_ranges(selection, q, store_batch_size=self.store_batch_size,
                                   query_batch_size=self.query_batch_size, 
                                   dim_batch_size=self.dim_batch_size)
        is_in_range = mask.any(dim=1) # (n, k) -> (n,)
        found_ids, = torch.where(is_in_range)
        del selection, mask, is_in_range
        return remap_ids(found_ids, ids)
    
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
        This is default implementation which is heavy inefficient O(N), N - number of semantics.
    '''
    
    def _insert_distinct(self, unique_vectors: torch.Tensor) -> list[int]:
        found_ids = self.query_points(unique_vectors)
        missing = get_missing_ids(found_ids)
        if len(missing) > 0:
            missing_ids = self._alloc_vectors(unique_vectors[missing])
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
                max_children: int = math.inf, 
                switch_to_all_cap = 0.5, **kwargs):
        ''' 
            epsilon: Size of the bin in each dimension (0 or 1 dim tensor)
            max_bin_size: Maximum number of elements in a bin, if set, 
                            grid resize (expensive) will be triggered with new epsilon that 
                            would satisfy this condition.
        '''
        super().__init__(**kwargs)
        # self.epsilons = epsilons
        self.max_bin_size = max_children
        self.switch_to_all_cap = switch_to_all_cap
        self.bins: dict[tuple, list[int]] = {} # tuple is bin index

    def get_bin_index(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Should be defined by concrete index 
            vectors shape (n, dims)
        '''
        pass

    # def get_closest_bin_indices(self, vectors: torch.Tensor, obj_ids: torch.Tensor | None = None) -> torch.Tensor:
    #     pass 

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
        print(f"\tRebuild: {end_r_time - start_r_time}. Sz: {len(self.bins[trigger_bin_id])} Iter: {iters}") #Eps: {min(self.epsilons.tolist())}:{max(self.epsilons.tolist())}")     
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
            found_ids = self._alloc_vectors(unique_vectors)
        else:
            trigger_bin_id = None
            all_bin_entries = [e for bin_index in self.bins.keys() for e in self.bins.get(bin_index, [])]
            if len(all_bin_entries) > self.switch_to_all_cap * self.cur_id:
                all_bin_entries = None            
            found_ids = self.find_close(all_bin_entries, unique_vectors) # recompute stats if needed
            missing_ids = get_missing_ids(found_ids)
            if len(missing_ids) > 0:
                new_ids = self._alloc_vectors(unique_vectors[missing_ids])
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
    
    def query_points(self, query: torch.Tensor, 
                     atol: float | None = None, rtol: float | None = None) -> list[int]:
        ''' O(1) '''
        bin_indices = self.get_bin_index(query)
        bin_ids = set([tuple(row) for row in bin_indices.tolist()])
        # for qid, bin_id in enumerate(bin_indices.tolist()):
        #       bin_ids.setdefault(tuple(bin_id), []).append(qid)
        del bin_indices
        bin_entries = [e for bin_index in bin_ids for e in self.bins.get(bin_index, [])]
        if len(bin_entries) > self.switch_to_all_cap * self.cur_id:
            bin_entries = None
        found_ids = self.find_close(bin_entries, query, atol = atol, rtol = rtol) # recompute stats if needed
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
    
    # def query_closest(self, query: torch.Tensor, obj_ids: torch.Tensor | None = None) -> list[int]:
    #     bin_indices = self.get_closest_bin_indices(query, obj_ids)
    #     bin_ids = set([tuple(row) for row in bin_indices.tolist()])
    #     del bin_indices
    #     bin_entries = [e for bin_index in bin_ids for e in self.bins.get(bin_index, [])]
    #     if len(bin_entries) > self.switch_to_all_cap * self.cur_id:
    #         bin_entries = None
    #     found_ids = self.find_closest(bin_entries, query, obj_ids)
    #     return found_ids
    
    def get_bins_range(self, qrange: torch.Tensor):
        bin_qrange = self.get_bin_index(qrange)
        all_bin_ids_list = [bin_id for bin_id in self.bins.keys()]
        all_bin_ids = torch.tensor(all_bin_ids_list, dtype = torch.int64, device=qrange.device)
        bin_id_ids = find_in_range(all_bin_ids, bin_qrange)
        selected_bin_ids = [all_bin_ids_list[i] for i in bin_id_ids.tolist()]
        del bin_id_ids, all_bin_ids, bin_qrange
        return selected_bin_ids, qrange

    def query_range(self, qrange: torch.Tensor) -> list[int]:
        ''' Assumes that mapping to bin space is continuous: lines are left lines '''
        selected_bin_ids, qrange = self.get_bins_range(qrange)
        # bin_ranges = product(*[list(range(min(b1, b2), max(b1, b2) + 1)) for b1, b2 in zip(min_bin, max_bin)])
        range_entries = [e for bin_index in selected_bin_ids for e in self.bins.get(bin_index, [])]
        if len(range_entries) == 0:
            return []        
        if qrange is None:
            return range_entries
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

    def get_bin_index(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Get bin index for a given vector '''
        return torch.floor(vectors // self.epsilons).to(dtype=torch.int64)
    
    # def get_closest_bin_indices(self, vectors: torch.Tensor, obj_ids: torch.Tensor | None = None) -> torch.Tensor:
    #     pass
    
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
        
# class MBR:        

#     def __init__(self, mbr: torch.Tensor, points: Optional[torch.Tensor] = None):
#         ''' mbr shape (2, *k, dims), points shape (n, *k, dims) '''
#         self.mbr = mbr
#         self.points = points
#         self.point_ids: Optional[list[int]] = None
#         self._area: Optional[float] = None 
            
#     def has_points(self) -> bool:
#         return self.points is not None

#     def get_min(self) -> torch.Tensor:
#         return self.mbr[0]
    
#     def get_max(self) -> torch.Tensor:
#         return self.mbr[1]
        
#     def area(self) -> float:        
#         if self._area is None:
#             self._area = torch.prod(self.get_max() - self.get_min()).item()
#         return self._area
    
#     def enlarge(self, *p: "MBR") -> "MBR":
#         if len(p) == 0:
#             return self
#         mbrs = torch.stack([self.mbr, *(x.mbr for x in p)])
#         new_mbr = torch.empty((2, *mbrs.shape[1:]), dtype=mbrs.dtype, device=mbrs.device)
#         new_mbr[0] = mbrs[:, 0].min(dim=0).values
#         new_mbr[1] = mbrs[:, 1].max(dim=0).values
#         new_mbr = MBR(mbrs) # do not pass points here 
#         return new_mbr
    
#     def enlargement(self, p: "MBR") -> tuple[float, float, "MBR"]:
#         ''' Resize MBR to cover the point.
#         '''
#         new_mbr = self.enlarge(p)
#         new_area = new_mbr.area()
#         old_area = self.area()
#         return new_area - old_area, old_area, new_mbr     
    

# def intersect_points(points: torch.Tensor, mbr2: torch.Tensor) -> torch.Tensor:
#     ''' points (n, dims), mbr2 (2, m, dims) 
#         returns mask (n, m) - 1 point n is in range m
#     '''
#     res = torch.all((points.unsqueeze(1) <= mbr2[1].unsqueeze(0)) & (mbr2[0].unsqueeze(0) <= points.unsqueeze(1)), dim=-1) # (n, m)
#     return res

@dataclass(eq=False, unsafe_hash=False)
class RTreeNode: 

    def get_child_mbrs(self, as_tuple: bool = False, **kwargs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def is_leaf(self) -> bool:
        return True 
    
    def get_size(self):
        return len(self.children)

    def get_new_node(self, selected_ids: list[int], **kwargs) -> tuple[torch.Tensor, "RTreeNode"]:
        pass
    
@dataclass(eq=False, unsafe_hash=False)
class RTreeBranch(RTreeNode):
    mbrs: torch.Tensor
    ''' mbr of shape [2, max_children, dims] 
        mbr[0] - mins, mbr[1] - maxs
    '''
    children: list["RTreeNode"]

    def get_child_mbrs(self, as_tuple: bool = False, **kwargs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        res = self.mbrs[:,:len(self.children)]
        if as_tuple:
            return (res[0], res[1])
        return res
    
    def is_leaf(self) -> bool:
        return False 
    
    def set_child(self, child_id, mbr: torch.Tensor, node: "RTreeNode"):
        self.mbrs[:, child_id] = mbr
        self.children[child_id] = node
    
    def add_children(self, new_children: list[tuple[torch.Tensor, "RTreeNode"]]):
        if len(new_children) == 0:
            return
        new_size = self.get_size() + len(new_children)
        if new_size > self.mbrs.shape[1]: #
            old_mbrs = self.mbrs
            self.mbrs = torch.zeros((2, new_size, self.mbrs.shape[2]), dtype=self.mbrs.dtype, device=self.mbrs.device)
            self.mbrs[:, :old_mbrs.shape[1]] = old_mbrs
        for i in range(len(new_children)):
            mbr, node = new_children[i]
            self.mbrs[:, len(self.children)] = mbr
            self.children.append(node)
    
    def get_new_node(self, selected_ids: list[int], **kwargs) -> tuple[torch.Tensor, "RTreeNode"]:
        new_mbrs = self.mbrs.clone()
        new_mbrs[:, :len(selected_ids)] = self.mbrs[:, selected_ids]
        new_children = [self.children[i] for i in selected_ids]
        mbr = torch.zeros((2, self.mbrs.shape[2]), dtype=self.mbrs.dtype, device=self.mbrs.device)
        mbr[0] = torch.min(new_mbrs[0, :len(selected_ids)], dim=0).values
        mbr[1] = torch.max(new_mbrs[1, :len(selected_ids)], dim=0).values
        # mbr[2] = torch.mean(new_mbrs[2, :len(selected_ids)], dim=0)
        return (mbr, RTreeBranch(new_mbrs, list(new_children)))
    
@dataclass(eq=False, unsafe_hash=False)
class RTreeLeaf(RTreeNode):
    children: list[int]
    
    def get_new_node(self, selected_ids: list[int], *, store: VectorStorage, **kwargs) -> tuple[torch.Tensor, "RTreeNode"]:
        new_ids = [self.children[i] for i in selected_ids]
        vectors = store.get_vectors(new_ids)
        mbr = torch.zeros((2, vectors.shape[-1]), dtype=vectors.dtype, device=vectors.device)
        mbr[0] = torch.min(vectors, dim=0).values
        mbr[1] = torch.max(vectors, dim=0).values
        # mbr[2] = torch.mean(vectors, dim=0)
        return mbr, RTreeLeaf(new_ids)
    
    def get_child_mbrs(self, as_tuple = False, *, store: VectorStorage, **kwargs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        selection_ids = self.children
        vectors = store.get_vectors(selection_ids)
        if as_tuple:
            return (vectors, vectors)
        res = torch.stack((vectors, vectors), dim=0) # (2, n, dims)
        return res        

class RTreeIndex(SpatialIndex):
    ''' R-Tree spatial index for NN search.
        It is a tree structure where each node contains a minimum bounding rectangle (MBR) that covers its children.
        The MBR is defined by the minimum and maximum coordinates in each dimension.

        Notes:
        1. Error with insertion: if 2 mbrs have minimal enlargement, there is possiblity to pick incorrect one
            that does not have existing point - unnecessary duplicate allocation.
        2. In many cases it is slower that grid even with grid rebuild. 
    '''
    def __init__(self, max_children: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_children = max_children
        self.root = RTreeLeaf([])
        self.root_mbr = None

    def split(self, node_mbr: torch.Tensor, node: RTreeNode, split_iter = 0) -> list[tuple[torch.Tensor, RTreeNode]]:
                        # overflow: RTreeBranch | list[int]) -> list[RTreeNode]:))
        if node.get_size() <= self.max_children:
            return []
        # print(f"Splitting node with {node.get_size()} children. Iter {split_iter}")
        min_mbrs, max_mbrs = node.get_child_mbrs(as_tuple = True, store=self)   
        # assert torch.all((min_mbrs <= max_mbrs)), "Min MBRs should be less than or equal to Max MBRs"     
        # assert torch.all( (min_mbrs <= node_mbr[1]) & (node_mbr[0] <= min_mbrs)), "Node Min MBRs should cover child MBRs"
        # assert torch.all( (max_mbrs <= node_mbr[1]) & (node_mbr[0] <= max_mbrs)), "Node Min MBRs should cover child MBRs"
        center_mbrs = (min_mbrs + max_mbrs) / 2

        # children_groups = torch.zeros((center_mbrs.shape[0], ), dtype=torch.uint8, device=center_mbrs.device) # (n, 2) - left and right children
        def split_rec(cur_ids: Optional[torch.Tensor] = None):
            if cur_ids is None:
                cur_center_mbrs = center_mbrs
                cur_min_mbrs = min_mbrs
                cur_max_mbrs = max_mbrs
            else:
                if len(cur_ids) < self.max_children:
                    l = cur_ids.tolist()
                    del cur_ids
                    return [l]
                cur_center_mbrs = center_mbrs[cur_ids]
                cur_min_mbrs = min_mbrs[cur_ids]
                cur_max_mbrs = max_mbrs[cur_ids]
            cur_median = torch.median(cur_center_mbrs, dim=0).values # (dims,)
            cur_median_mask = (cur_min_mbrs <= cur_median) & (cur_median <= cur_max_mbrs) # (n, dims), median intersection mask
            cur_dim_scores = torch.sum(cur_median_mask, dim=0) # (dims,) - how many children intersect median
            cur_best_dim = torch.argmin(cur_dim_scores)
            child_split_mask = cur_center_mbrs[:, cur_best_dim] <= cur_median[cur_best_dim]
            new_ids1, = torch.where(child_split_mask)
            child_split_mask.logical_not_()
            new_ids2, = torch.where(child_split_mask)
            if cur_ids is not None:
                new_ids1 = cur_ids[new_ids1]
                new_ids2 = cur_ids[new_ids2]
            del cur_center_mbrs, cur_min_mbrs, cur_max_mbrs, cur_median, cur_median_mask, cur_dim_scores
            del cur_best_dim, child_split_mask
            left_split = split_rec(new_ids1)
            right_split = split_rec(new_ids2)
            return [*left_split, *right_split]

        children_groups = split_rec()
        nodes = []
        for child_ids in children_groups:
            child_mbr, child_node = node.get_new_node(child_ids, store = self)
            # child_child_mins, child_child_maxs = child_node.get_child_mbrs(as_tuple=True, store=self)
            # assert torch.all((child_mbr[0] <= child_child_mins) & (child_mbr[1] >= child_child_maxs)), "Child MBRs should be valid"
            nodes.append((child_mbr, child_node))
        return nodes

        # min_dist_mask = torch.isclose(max_mbrs, node_mbr[0], rtol = self.rtol, atol = self.atol)
        # min_dist_mask.neg_()
        # min_dist_counts = min_dist_mask.sum(dim=-1)
        # distinct_dist_counts = torch.unique(min_dist_counts, return_inverse=False)
        # min_dists = max_mbrs - node_mbr[0]
        # max_dists = node_mbr[1] - min_mbrs
        # distinct_dist_count_ids = torch.argsort(distinct_dist_counts)
        # for dist_count_id in distinct_dist_count_ids:
        #     dist_count = distinct_dist_counts[dist_count_id]
        #     selected_ids, = torch.where(min_dist_counts == dist_count)
        #     selected_mask = min_dist_mask[selected_ids]
        #     selelcted_min_dists = torch.masked_select(min_dists[selected_ids], selected_mask).view(-1, dist_count)
        #     # selelcted_max_dists = torch.masked_select(max_dists[selected_ids], selected_mask).view(-1, dist_count)
        # min_areas = torch.prod(max_mbrs - node_mbr[0], dim=-1) #enlargements of children 
        # max_areas = torch.prod(node_mbr[1] - min_mbrs, dim=-1) # areas of children
        # child_ids = torch.argsort(min_areas)
        # for child_id in child_ids:
        #     min_area = min_areas[child_id]
        #     max_area = max_areas[child_id]
        #     bin_id = 0 if min_area <= max_area else 1
        #     children[bin_id].append(child_id.item())
        # nodes = [node.get_new_node(child_ids) for child_ids in children]
        # return nodes        

    # NOTE: PROBLEM: when 2 mbrs have both min extensions - one could contain point, another - not,
    #       this will allocate unnecessary duplicate in the store if incorrect mbr is selected
    #       problem is that querying and insertion have different procedures 
    def _insert(self, node_mbr: torch.Tensor, node: RTreeNode, points: torch.Tensor, 
                            found_ids: list[int], point_ids: Optional[torch.Tensor] = None) -> list[tuple[torch.Tensor, RTreeNode]]:
        ''' Propagates points through tree splitting point_ids 
        '''
        cur_points = points[point_ids] if point_ids is not None else points
        cur_mins = torch.min(cur_points, dim = 0).values
        cur_maxs = torch.max(cur_points, dim = 0).values
        torch.minimum(node_mbr[0], cur_mins, out = node_mbr[0])
        torch.maximum(node_mbr[1], cur_maxs, out = node_mbr[1])
        del cur_mins, cur_maxs
        if node.is_leaf():
            # point_ids_list = point_ids.tolist()
            # if point_ids:
            #     del point_ids

            # yield (node_mbr, node, cur_points)

            if node.get_size() > 0:
                local_found_ids = self.find_close(node.children, cur_points)
            else:
                local_found_ids = [-1] * cur_points.shape[0]
            missing_ids = get_missing_ids(local_found_ids)
            new_ids = []
            if len(missing_ids) > 0:
                points_missing_tensor = cur_points[missing_ids]
                new_ids = self._alloc_vectors(points_missing_tensor)
                local_found_ids = merge_ids(local_found_ids, new_ids)
                node.children.extend(new_ids)
            if point_ids is None:
                for local_id, local_id_value in enumerate(local_found_ids):
                    found_ids[local_id] = local_id_value
            else:
                for local_id, local_id_value in enumerate(local_found_ids):
                    found_ids[point_ids[local_id].item()] = local_id_value
            replacement = self.split(node_mbr, node)
            return replacement

        child_mbrs = node.get_child_mbrs()

        cur_diag = child_mbrs[1] - child_mbrs[0] # (n, dims)
        new_mins = torch.min(cur_points.unsqueeze(1), child_mbrs[0].unsqueeze(0)) # (m, n, dims)
        new_maxs = torch.max(cur_points.unsqueeze(1), child_mbrs[1].unsqueeze(0)) # (m, n, dims)
        diag_delta = (new_maxs - new_mins) - cur_diag # (m, n, dims)
        diag_changes = torch.norm(diag_delta, dim=-1) # (m, n)
        point_child_ids = torch.argmin(diag_changes, dim=-1) # (m, ) - per point closes children
        next_children = []
        for child_id in range(child_mbrs.shape[1]):
            local_child_point_ids, = torch.where(point_child_ids == child_id)
            if local_child_point_ids.numel() == 0:
                continue
            if point_ids is None:
                child_point_ids = local_child_point_ids
            else:
                child_point_ids = point_ids[local_child_point_ids]
            del local_child_point_ids
            if child_point_ids.numel() == points.shape[0]:
                del child_point_ids
                child_point_ids = None
            next_children.append((child_id, child_mbrs[:,child_id], child_point_ids))

        del cur_points, cur_diag, new_mins, new_maxs, diag_delta, diag_changes, point_child_ids

        additional_children = []
        for child_id, child_mbr, child_point_ids in next_children:
            child_replacement = self._insert(child_mbr, node.children[child_id], points, found_ids, child_point_ids)
            if len(child_replacement) > 0:
                node.set_child(child_id, child_replacement[0][0], child_replacement[0][1])
                for i in range(1, len(child_replacement)):
                    additional_children.append(child_replacement[i])
        node.add_children(additional_children)
        replacement = self.split(node_mbr, node)
        return replacement

    def _insert_distinct(self, unique_vectors: torch.Tensor) -> list[int]:
        ''' Inserts one point (rects are not supported yet) ''' 
        if self.root_mbr is None:
            self.root_mbr = torch.zeros((2, unique_vectors.shape[-1]), dtype=unique_vectors.dtype, device=unique_vectors.device)
            self.root_mbr[0] = unique_vectors.min(dim=0).values
            self.root_mbr[1] = unique_vectors.max(dim=0).values
        # else:
        #     torch.minimum(self.root_mbr[0], unique_vectors.min(dim=0).values, out = self.root_mbr[0])
        #     torch.maximum(self.root_mbr[1], unique_vectors.max(dim=0).values, out = self.root_mbr[1])

        found_ids = [-1] * unique_vectors.shape[0] # -1 means not found
        root_replacement = self._insert(self.root_mbr, self.root, unique_vectors, found_ids) 
        while len(root_replacement) > 0:
            child_mbrs, children = zip(*root_replacement)
            child_tensor = torch.empty((2, max(self.max_children, len(child_mbrs)), unique_vectors.shape[-1]), dtype=unique_vectors.dtype, device=unique_vectors.device)
            for mbr_id, mbr in enumerate(child_mbrs):
                child_tensor[:, mbr_id] = mbr
            self.root = RTreeBranch(child_tensor, list(children)) 
            root_replacement = self.split(self.root_mbr, self.root)            
        return found_ids  

    def _query_point_ids(self, node: RTreeNode, points: torch.Tensor) -> Generator[int, None, None]:
        if node.is_leaf():
            yield from node.children
        else:
            child_mbrs = node.get_child_mbrs()
            in_range_mask = find_in_ranges(points, child_mbrs, store_batch_size=self.store_batch_size,
                                            query_batch_size=self.query_batch_size,
                                            dim_batch_size=self.dim_batch_size) # (n, m)
            # in_range_mask_t = in_range_mask.t()
            any_point_mask = in_range_mask.any(dim=0)  # (n, m) -> (m, )
            child_ids, = torch.where(any_point_mask)
            for child_id in child_ids.tolist():
                point_ids, = torch.where(in_range_mask[:, child_id])
                yield from self._query_point_ids(node.children[child_id], points[point_ids])
                del point_ids
            del in_range_mask, any_point_mask, child_ids

    def query_points(self, query: torch.Tensor,
                     atol: float | None = None, rtol: float | None = None) -> list[int]:
        point_ids = list(self._query_point_ids(self.root, query))
        if len(point_ids) == 0:
            return [-1] * query.shape[0]
        found_ids = self.find_close(point_ids, query, atol = atol, rtol = rtol)
        return found_ids
    
    def _query_range_ids(self, node: RTreeNode, mbrs: torch.Tensor) -> Generator[int, None, None]:
        if node.is_leaf():
            yield from node.children
        else:
            child_mbrs = node.get_child_mbrs()
            int_mask = find_intersects(child_mbrs, mbrs, store_batch_size=self.store_batch_size,
                                   query_batch_size=self.query_batch_size, 
                                   dim_batch_size=self.dim_batch_size)
            any_range_mask = int_mask.any(dim=1) # (n, m) -> (n, )
            child_ids, = torch.where(any_range_mask)
            for child_id in child_ids.tolist():
                mbr_ids, = torch.where(int_mask[child_id])
                yield from self._query_range_ids(node.children[child_id], mbrs[:, mbr_ids])
                del mbr_ids
            del int_mask, any_range_mask, child_ids       
    
    def query_range(self, qrange: torch.Tensor) -> list[int]:
        ''' qrange shape (2, dims) --> (2, 1, dims) or (2, k, dims) '''
        if qrange.shape[0] == 2:
            qrange = qrange.unsqueeze(1)  # (2, 1, dims)
        point_ids = list(self._query_range_ids(self.root, qrange))
        if len(point_ids) == 0:
            return []
        found_ids = self.find_in_ranges(point_ids, qrange)
        return found_ids
    
    def _get_mbrs(self, node_mbr: torch.Tensor, node: RTreeNode):
        ''' Returns MBRs of the node '''
        if node.is_leaf():
            yield node_mbr
        else:
            child_mbrs = node.get_child_mbrs()
            for child_id in range(child_mbrs.shape[1]):
                yield from self._get_mbrs(child_mbrs[:, child_id], node.children[child_id])

    def get_mbrs(self):
        return list(self._get_mbrs(self.root_mbr, self.root))

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
    def __init__(self, target: torch.Tensor, pack_dtype = torch.int64, **kwargs):
        super().__init__(**kwargs)
        self.epsilons: torch.Tensor = torch.zeros((self.vectors.shape[-1], ), dtype=self.vectors.dtype, device=self.vectors.device)
        sz = torch.iinfo(pack_dtype).bits
        self.int_dims = math.ceil(self.vectors.shape[-1] / sz)
        # self.interactions = torch.zeros((self.capacity, int_dims), dtype=pack_dtype, device=self.vectors.device)
        self.target = target
        self.pack_dtype = pack_dtype
        self.best_interactions = self.get_bin_index(torch.zeros((self.vectors.shape[-1], ), dtype=self.vectors.dtype, device=self.vectors.device), is_distance=True)
        pass
        # self.iid_to_vids: dict[int, list[int]] = {} # interaction id to vector ids
        # self.vid_to_iid: dict[int, int] = {} # vector id to interaction id
    
    def get_bin_index(self, vectors: torch.Tensor, is_distance = False) -> list[tuple]:
        if is_distance:
            assert torch.all(vectors >= 0)
            distances = vectors 
        else:
            distances = torch.abs(vectors - self.target)
        interactions = (distances <= self.epsilons)
        ints = pack_bits(interactions, dtype = self.pack_dtype)
        del interactions, distances
        return ints    
    
    def on_rebuild(self, trigger_bin_id: tuple):
        # vectors = self.get_vectors(self.bins[trigger_bin_id])
        # distances = torch.abs(vectors - self.target)
        # new_epsilons = distances.mean(dim=0)
        # zero = torch.tensor(0, dtype=new_epsilons.dtype, device=new_epsilons.device)
        # self.epsilons = torch.where(torch.isclose(new_epsilons, zero, atol=self.atol, rtol=self.rtol), self.epsilons, new_epsilons)
        # del vectors, distances

        vectors = self.get_vectors(self.bins[trigger_bin_id]) # get all vectors in the bin
        distances = torch.abs(vectors - self.target)
        medians = distances.median(dim = 0).values
        self.epsilons[:] = medians
        # balances = (distances - medians).sum(dim=0)
        # balances.abs_()
        # sort_ids = torch.argsort(balances)
        # approx_num_dims = math.floor(math.log2(distances.shape[0] / self.max_bin_size)) + 1
        # selected_dims = sort_ids[:approx_num_dims] # take only first approx_num_dims dimensions
        # self.epsilons[selected_dims] = medians[selected_dims]

    def get_bins_range(self, qrange: torch.Tensor):
        ''' qrange is 1d distance tensor (dims), distance from target '''
        distant_bin = self.get_bin_index(qrange, is_distance=True)        
        all_bin_ids_list = [bin_id for bin_id in self.bins.keys()]
        all_bin_ids = torch.tensor(all_bin_ids_list, dtype = torch.int64, device=qrange.device)
        not_equal_bits = (all_bin_ids ^ distant_bin) | (all_bin_ids ^ self.best_interactions)
        not_equal_bits.bitwise_not_() # invert bit
        found_ids, = torch.where(torch.all(not_equal_bits == 0, dim=-1))
        selected_bin_ids = all_bin_ids[found_ids]
        selected_bin_ids_list = [tuple(bin_id) for bin_id in selected_bin_ids.tolist()]
        del not_equal_bits, all_bin_ids, found_ids
        new_qrange = torch.stack((self.target - qrange, self.target + qrange), dim=0) # (2, dims)
        return selected_bin_ids_list, new_qrange
    
def get_cos_distance(v1: torch.Tensor, v2: torch.Tensor, v1_norms = None, v2_norms = None,
                        rtol = 1e-5, atol=1e-4, zero=None) -> torch.Tensor:
    if v1_norms is None:
        v1_norms = torch.norm(v1, dim=-1)
    if v2_norms is None:
        v2_norms = torch.norm(v2, dim=-1)
    norm_prod = v1_norms * v2_norms
    if zero is None:
        zero = torch.zeros(1, dtype=v1_norms.dtype, device=v1_norms.device)
    cos_distance = 1 - torch.sum(v1 * v2, dim=-1) / norm_prod
    zero_ids, = torch.where(torch.isclose(norm_prod, zero, atol = atol, rtol=rtol))
    cos_distance[zero_ids] = zero
    return cos_distance

    
class RCosIndex(BinIndex):
    ''' Represents torch.Tensor with only radius vector and cosine distance to target vector 
        Splits spalce onto cones by angle and radius.

        Note: does not work with float16, as too many points land into same range and angle region --> splits epsilon till zeroes
    ''' 
    def __init__(self, target: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        if not torch.is_tensor(target):
            target = torch.full((self.vectors.shape[-1], ), target, dtype=self.vectors.dtype, device=self.vectors.device)
        self.target = target
        self.target_norm = torch.norm(self.target)
        self.zero = torch.zeros_like(self.target_norm)
        assert not torch.isclose(self.target_norm, self.zero, atol=self.atol, rtol=self.rtol), "Target vector cannot be zero vector."
        # if torch.is_tensor(epsilons):
        #     self.epsilons = epsilons
        # else:
        self.epsilons = torch.tensor([1, 1], dtype=self.vectors.dtype, device=self.vectors.device)

    def get_bin_index(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Get bin index for a given vector '''
        norms = torch.norm(vectors, dim=-1)
        cos_distance = get_cos_distance(vectors, self.target, v1_norms=norms, v2_norms=self.target_norm, 
                                        rtol=self.rtol, atol=self.atol, zero=self.zero)
        norm_bin_index = torch.floor(norms / self.epsilons[0]).to(dtype=torch.int64)
        cos_bin_index = torch.floor(cos_distance / self.epsilons[1]).to(dtype=torch.int64)
        bin_index = torch.stack((norm_bin_index, cos_bin_index), dim=-1) # shape (n, 2)
        del norms, cos_distance
        return bin_index
    
    def on_rebuild(self, trigger_bin_id: tuple):
        # bin_tensor = self.get_vectors(self.bins[trigger_bin_id])
        # norms = torch.norm(bin_tensor, dim=1)
        # cos_distance = get_cos_distance(bin_tensor, self.target, v1_norms=norms, v2_norms=self.target_norm,
        #                                 rtol=self.rtol, atol=self.atol, zero=self.zero)

        approx_num_dims = math.floor(math.log2(len(self.bins[trigger_bin_id]) / self.max_bin_size)) + 1

        nums_dims = math.ceil(approx_num_dims / 2)

        self.epsilons /= 2 ** nums_dims 
        pass

        # norms_median = norms.median()
        # norms_start = trigger_bin_id[0] * self.epsilons[0]
        # new_norm_epsilon = torch.maximum(norms_median - norms_start, norms_start + self.epsilons[0] - norms_median)
        # if new_norm_epsilon <= 0:
        #    new_norm_epsilon = self.epsilons[0] / 2 
        # cos_median = cos_distance.median()
        # cos_start = trigger_bin_id[1] * self.epsilons[1]
        # new_cos_epsilon = torch.maximum(cos_median - cos_start, cos_start + self.epsilons[1] - cos_median)
        # if new_cos_epsilon <= 0:
        #     new_cos_epsilon = self.epsilons[1] / 2
        # self.epsilons = torch.tensor([new_norm_epsilon, new_cos_epsilon], dtype=self.vectors.dtype, device=self.vectors.device)
        pass
        # epsilon_ids, = torch.where(~torch.isclose(new_epsilon, self.zero, atol=self.atol, rtol=self.rtol))
        # self.epsilons[epsilon_ids] = new_epsilon[epsilon_ids]

    def get_bins_range(self, qrange):
        ''' qrange is 2 reference vectors that specify radius and angle from target '''
        bin_qrange = self.get_bin_index(qrange)
        bin_qrange[:] = torch.sort(bin_qrange, dim=0).values
        all_bin_ids_list = [bin_id for bin_id in self.bins.keys()]
        all_bin_ids = torch.tensor(all_bin_ids_list, dtype = torch.int64, device=qrange.device)
        bin_id_ids = find_in_range(all_bin_ids, bin_qrange)
        selected_bin_ids = [all_bin_ids_list[i] for i in bin_id_ids.tolist()]
        del bin_id_ids, all_bin_ids, bin_qrange
        return selected_bin_ids, None        

        
def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_rank = torch.argsort(torch.argsort(x)).to(dtype=x.dtype)
    y_rank = torch.argsort(torch.argsort(y)).to(dtype=y.dtype)

    # Compute Pearson correlation on the ranks
    x_mean = x_rank.mean(dim=-1).unsqueeze(-1)  # Ensure x_mean is a column vector
    y_mean = y_rank.mean(dim=-1).unsqueeze(-1)  # Ensure y_mean is a column vector
    numerator = torch.sum((x_rank - x_mean) * (y_rank - y_mean), dim=-1)
    denominator = torch.sqrt(torch.sum((x_rank - x_mean)**2) * torch.sum((y_rank - y_mean)**2, dim=-1))

    cor = numerator / denominator
    return cor
        
class SpearmanCorIndex(BinIndex):
    ''' vector to spearman correlation with target 
    
        NOTE: reduction of dims to 1 leads to very small epsilon, machine error and fails due to float precision.
        Use higher number of children is required here: max_children = 1000-10000,
        Works only with float32 (see RCosIndex problem)

        NOTE: very slow - same to absent index situation, 
             Many points land to small region.
        FIXME: ??epsilon should not be constant, starting from small value it should increase
               ??Do not hard fix max_children for this index - make it relaxed - max children only in near target region
    ''' 

    def __init__(self, target: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1
        if not isinstance(target, torch.Tensor):
            self.target = torch.full((self.vectors.shape[-1], ), target, dtype=self.vectors.dtype, device=self.vectors.device)
        else:
            self.target = target
    
    def get_bin_index(self, vectors: torch.Tensor) -> torch.Tensor:
        cors: torch.Tensor = 1 - spearman_correlation(vectors, self.target)
        bin_id = torch.floor(cors / self.epsilon).to(dtype=torch.int64)
        return bin_id.unsqueeze(-1)
    
    def on_rebuild(self, trigger_bin_id: tuple):
        # bin_tensor = self.get_vectors(self.bins[trigger_bin_id])

        approx_num_dims = math.floor(math.log2(len(self.bins[trigger_bin_id]) / self.max_bin_size)) + 1

        nums_dims = math.ceil(approx_num_dims / 2)

        self.epsilon /= 2 ** nums_dims 

        # cors: torch.Tensor = spearman_correlation(bin_tensor, self.target)
        # cors.abs_()
        # cor_start = trigger_bin_id[0] * self.epsilon
        # cor_end = (trigger_bin_id[0] + 1) * self.epsilon
        # cor_median = torch.median(cors).item()
        # self.epsilon = max(cor_end - cor_median, cor_median - cor_start)
        pass

    def get_bins_range(self, qrange):
        ''' qrange is distance in cor index - 2d tensor (1, 2), start-end of expected cor '''
        cor_delta = torch.floor(qrange / self.epsilon).to(dtype=torch.int64)
        min_bin_id = (cor_delta[0, 0].item(), )
        max_bin_id = (cor_delta[1, 0].item(), )
        selected_bin_ids = [bin_id for bin_id in self.bins.keys() if min_bin_id <= bin_id <= max_bin_id]
        return selected_bin_ids, None

def test_storage(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cpu"):
    storage = VectorStorage(capacity, dims, dtype=dtype, device=device)
    all_ids = []
    all_chunks = []
    for chunk_sz in range(1, 10):
        chunk = torch.randn((chunk_sz, dims), dtype=dtype, device=device)
        ids = storage._alloc_vectors(chunk)
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
                        num_points_per_group = 100, num_groups = 101, index = SpatialIndex,
                        max_children = 64):
    
    idx = index(capacity=capacity, dims=dims, dtype=dtype, device=device,
                        store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                        dim_batch_size = dim_batch_size, max_children = max_children)
    # start_insert_time = time.time()
    # idx.insert(torch.rand((capacity, dims), dtype = dtype, device=device))
    # for bin_id, point_ids in idx.bins.items():
    #     point_tensor = idx.get_vectors(point_ids)
    #     bin_tensor = torch.tensor(bin_id, dtype=idx.epsilons.dtype, device=idx.epsilons.device)
    #     bin_min = bin_tensor * idx.epsilons
    #     bin_max = bin_min + idx.epsilons
    #     assert torch.all((point_tensor >= bin_min) & (point_tensor <= bin_max)), "Grid index did not insert points in correct bins."    
    #     pass
    # print(f"Inserted {idx.cur_id:06} points in {start_time - start_insert_time:06.2f} seconds")
    insert_time = 0
    query_time = 0
    times = []
    for i in range(num_groups):
        # ids_range = (0, (i + 1) * num_points_per_group)
        new_tensors = ((i * 0.25) % 1) + 0.25 * torch.rand((num_points_per_group, dims), dtype = dtype, device=device)
        start_insert_time = time.time()
        idx.insert(new_tensors)        
        end_insert_time = time.time()
        orig_ids = torch.randint(idx.cur_id, (num_points_per_group,)).tolist()
        orig_ids.sort()
        query_points = idx.get_vectors(orig_ids)
        start_query_time = time.time()
        found_ids = idx.query_points(query_points)
        end_query_time = time.time()
        # found_ids = []
        # for query_point in query_points:
        #     one_found_ids = idx.query_points(query_point.unsqueeze(0)) # query one point at a time
        #     found_ids.extend(one_found_ids)
        found_ids.sort()
        assert found_ids == orig_ids, f"Spatial index did not return correct ids for group"
        insert_time += end_insert_time - start_insert_time
        query_time += end_query_time - start_query_time
        total_time = insert_time + query_time

        times.append(total_time)
        if i % 10 == 0:
            print(f"Queried group {i + 1:03}/{num_groups} in {total_time:05.2f} [{insert_time:05.2f},{query_time:05.2f}] seconds, "
                f"total {idx.cur_id:06} points, {store_batch_size:04}:{query_batch_size:03}:{dim_batch_size:03}|{max_children:03}")

    return times
    # assert idx.cur_id == num_points_per_group * num_groups, "Spatial index did not allocate all vectors correctly."


def test_spatial_index_query2(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        num_points_per_group = 100, num_groups = 101, indices: dict = {"spatial": SpatialIndex}):
    
    import numpy as np
    
    idxs = {idx_name: index(capacity=capacity, dims=dims, dtype=dtype, device=device) 
                        for idx_name, index in indices.items()}
    insert_times = {n:0 for n in idxs.keys()}
    query_times = {n:0 for n in idxs.keys()}
    range_times = {n:0 for n in idxs.keys()}
    times = {n:[] for n in idxs.keys()}
    cur_id = 0
    for i in range(num_groups):
        cur_num_points = np.random.randint(num_points_per_group // 2, num_points_per_group + 1)
        means = torch.rand(dims, dtype=dtype, device=device)
        std = np.random.rand() * 0.05 + 0.025
        stds = torch.full((cur_num_points, dims), std, dtype=dtype, device=device)
        distr = torch.normal(mean = means, std = stds)   
        distr.clamp_(0, 1)     
        same_count = min(cur_id, np.random.randint(0, distr.shape[0]))
        selected_ids = None
        if same_count > 0:
            tmp_perm = torch.randperm(distr.shape[0], device=device)
            selected_ids = tmp_perm[:same_count].tolist()
            del tmp_perm
            tmp_perm = torch.randperm(cur_id, device=device)
            from_idx_ids = tmp_perm[:same_count].tolist()
            del tmp_perm
        new_cur_ids = []
        idxs_returned_ids = []
        for idx_name, idx in idxs.items():
            if not idx_name.startswith('rtree'):
                assert idx.cur_id == cur_id
            if selected_ids is not None:
                distr[selected_ids] = idx.vectors[from_idx_ids]
            start_insert_time = time.time()
            returned_ids = idx.insert(distr)   
            end_insert_time = time.time()
            idxs_returned_ids.append(returned_ids)     
            insert_time = end_insert_time - start_insert_time
            insert_times[idx_name] += insert_time
            if not idx_name.startswith('rtree'):
                new_cur_ids.append(idx.cur_id)
        assert np.all([x == new_cur_ids[0] for x in new_cur_ids])
        cur_id = new_cur_ids[0]

        #querying points
        cur_num_points2 = np.random.randint(1, num_points_per_group + 1)
        same_count = min(cur_id, np.random.randint(0, cur_num_points2))
        queries = torch.rand((cur_num_points2, dims), dtype=dtype, device=device)
        tmp_perm = torch.randperm(queries.shape[0], device=device)
        selected_ids = tmp_perm[:same_count].tolist()
        del tmp_perm
        tmp_perm = torch.randperm(cur_id, device=device)
        to_ids = tmp_perm[:same_count].tolist()
        del tmp_perm
        for idx_name, idx in idxs.items():
            idx_q = queries.clone()
            idx_q[selected_ids] = idx.get_vectors(to_ids)
            # assert idx.cur_id == cur_id
            start_query_time = time.time()
            found_ids = idx.query_points(idx_q)
            assert len(found_ids) == idx_q.shape[0]
            end_query_time = time.time()
            query_time = end_query_time - start_query_time
            query_times[idx_name] += query_time
            found_ids_set = set(found_ids)
            to_ids_set = set(to_ids)
            to_ids_set.add(-1)
            if not idx_name.startswith('rtree'):
                assert set.issubset(to_ids_set, found_ids_set), f"Spatial index {idx_name} did not return correct ids for group {i + 1:02}/{num_groups}"
            del idx_q

        for idx_name, idx in idxs.items():
            total_time = insert_times[idx_name] + query_times[idx_name]
            times[idx_name].append(total_time)
        if i % 10 == 0:
            print(f"{i + 1:03}/{num_groups}")
            for idx_name, idx in sorted(idxs.items(), key=lambda x: times[x[0]][-1], reverse=True):
                print(f"\t{idx_name:<10} in {times[idx_name][-1]:05.2f} [{insert_times[idx_name]:05.2f},{query_times[idx_name]:05.2f}] seconds, "
                    f"total {cur_id:06} points")

    return times
    # assert idx.cur_id == num_points_per_group * num_groups, "Spatial index did not allocate all vectors correctly."


def test_spatial_index_range(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=4096, query_batch_size=256, dim_batch_size=4,
                        num_points_per_group = 100, num_groups = 1000,
                        time_deltas = False, index = SpatialIndex):
    
    idx = index(capacity=capacity, dims=dims, dtype=dtype, device=device,
                        store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                        dim_batch_size = dim_batch_size)
    # idx.insert(torch.rand((capacity, dims), dtype = dtype, device=device))
    # for bin_id, point_ids in idx.bins.items():
    #     point_tensor = idx.get_vectors(point_ids)
    #     bin_tensor = torch.tensor(bin_id, dtype=idx.epsilons.dtype, device=idx.epsilons.device)
    #     bin_min = bin_tensor * idx.epsilons
    #     bin_max = bin_min + idx.epsilons
    #     assert torch.all((point_tensor >= bin_min) & (point_tensor <= bin_max)), "Grid index did not insert points in correct bins."    
    #     pass
    start_time = time.time()
    times = []
    for i in range(num_groups):
        # ids_range = (0, (i + 1) * num_points_per_group)
        new_tensors = ((i * 0.25) % 1) + 0.25 * torch.rand((num_points_per_group, dims), dtype = dtype, device=device)
        new_ids = idx.insert(new_tensors)        
        min_p = new_tensors.min(dim=0).values
        max_p = new_tensors.max(dim=0).values
        qrange = torch.stack((min_p, max_p), dim=0) # (2, dims)
        new_ids = set(new_ids)
        found_ids = idx.query_range(qrange)
        found_ids = set(found_ids)
        assert set.issubset(new_ids, found_ids), f"Spatial index did not return correct ids for group {i + 1:02}/{num_groups}"
        iter_end = time.time()
        total_duration = iter_end - start_time
        if time_deltas:
            start_time = iter_end
        times.append(total_duration)
        if i % 10 == 0:
            print(f"Queried group {i + 1:02}/{num_groups} in {total_duration:06.2f} seconds, "
                f"{store_batch_size:04}:{query_batch_size:03}:{dim_batch_size:03}")

    return times
    # assert idx.cur_id == num_points_per_group * num_groups, "Spatial index did not allocate all vectors correctly."


def test_time(f, **arg_combs):    
    from matplotlib import pyplot as plt

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

def test_time2(f, *arg_combs):    
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 6))
    for arg_comb in arg_combs:
        name = arg_comb.pop("_name")
        print(f"--- {name} ---")
        times = f(**arg_comb)
        plt.plot(times, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()    

def test_time_all(f, **kwargs):    
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 6))
    times = f(**kwargs)
    for time_name, xy in times.items():
        plt.plot(xy, label=time_name)
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()        

def test_grid_index(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8):
    idx = GridIndex(epsilons=1, max_children = 2000, capacity=capacity, dims=dims, dtype=dtype, device=device,
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

def test_rcos_index(capacity = 100_000, dims = 1024, dtype = torch.float32, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8):
    target = torch.full((dims,), 0.5, dtype=dtype, device=device)
    idx = RCosIndex(target, max_children = 10, capacity=capacity, dims=dims, dtype=dtype, device=device,
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
    
    qry0 = torch.ones_like(target, dtype=dtype, device=device)
    qry0[0] = 0
    qry1 = torch.zeros_like(target, dtype=dtype, device=device)
    qry1[0] = 1
    qry = torch.stack((qry0, qry1), dim=0) # (2, dims)
    r_ids = idx.query_range(qry)
    assert sorted(r_ids) == list(range(idx.cur_id)), "Grid index did not return all ids in range query."

    
    pass 

def test_scor_index(capacity = 100_000, dims = 1024, dtype = torch.float32, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8):
    target = torch.full((dims,), 0.5, dtype=dtype, device=device)
    idx = SpearmanCorIndex(target, max_children = 1000, capacity=capacity, dims=dims, dtype=dtype, device=device,
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
    
    qry = torch.tensor([[0], [2]], dtype=dtype, device=device)
    r_ids = idx.query_range(qry)
    assert sorted(r_ids) == list(range(idx.cur_id)), "Grid index did not return all ids in range query."

    
    pass 

def test_int_index(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8):
    target = torch.full((dims,), 0.5, dtype=dtype, device=device)
    idx = InteractionIndex(target, max_children = 10, capacity=capacity, dims=dims, dtype=dtype, device=device,
                    store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                    dim_batch_size = dim_batch_size)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids1 = idx.insert(p)
    print("Insert 1 done")
    p_ids2 = idx.insert(p)
    print("Insert 2 done")
    assert sorted(p_ids1) == sorted(p_ids2), "Grid index did not return same ids for same points."
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids3 = idx.insert(p) # should trigger rebuild
    print("Insert 3 done")
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids4 = idx.insert(p)
    print("Insert 4 done")
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids5 = idx.insert(p)
    print("Insert 5 done")

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
    
    r_ids = idx.query_range(torch.ones((dims,), dtype=dtype, device=device))
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
    import numpy as np
    plt.ion()
    plt.clf()
    # plt.figure(figsize=(10, 6))
    
    # Plot scattered points
    plt.scatter(x, y, color="black", s = 4)
    
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
    plt.pause(2)

# Example usage
# x = [1, 2, 3, 4, 5]
# y = [5, 4, 3, 2, 1]
# rects = [(1.5, 1.5, 2, 2), (3.5, 3.5, 1, 1)]
# epsilons = (1, 1)

# visualize_2d(x, y, rects=rects, epsilons=epsilons)
# pass

def viz_idx(idx: SpatialIndex):
    if hasattr(idx, 'epsilons'):
        epsilonx, epsilony = idx.epsilons.tolist()
    else:
        epsilonx, epsilony = 0.1, 0.1
    vectors = idx.get_vectors(None)
    if isinstance(idx, GridIndex):
        rects = [(bin_id_x * epsilonx, bin_id_y * epsilony, epsilonx, epsilony) for bin_id_x, bin_id_y in idx.bins.keys()]
    elif isinstance(idx, RTreeIndex):
        mbrs_list = idx.get_mbrs()
        rects = [(mbr[0,0].item(), mbr[0,1].item(), w[0].item(), w[1].item()) for mbr in mbrs_list for w in [mbr[1] - mbr[0]]]
    xs = vectors[:, 0].tolist()
    ys = vectors[:, 1].tolist()
    visualize_2d(xs, ys,
                 rects=rects,
                 epsilons=(epsilonx, epsilony),
                 xrange=(0, 1), yrange=(0, 1))
    
def test_idx_distr(capacity = 100_000, dims = 2, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8,
                        num_groups = 20, group_size=10, idxb = partial(GridIndex, epsilons=1, max_bin_size = 10)):
    idx = idxb(capacity=capacity, dims=dims, dtype=dtype, device=device,
                    store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                    dim_batch_size = dim_batch_size)
    for num_groups in range(num_groups):
        means = torch.rand(dims, dtype=dtype, device=device)
        stds = torch.full((group_size, dims), 0.05, dtype=dtype, device=device)
        distr = torch.normal(mean = means, std = stds)
        distr.clamp_(0, 1)
        # distr = shifts + torch.rand((100, dims), dtype=dtype, device=device) * 0.1
        idx.insert(distr)
        viz_idx(idx)    

    pass

def test_rtree_index(capacity = 100_000, dims = 2, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8,
                        num_points_per_group = 3, num_groups = 100):
    idx = RTreeIndex(max_children = 10, capacity=capacity, dims=dims, dtype=dtype, device=device,
                        store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                        dim_batch_size = dim_batch_size)
    for i in range(num_groups):
        p = torch.rand((num_points_per_group, dims), dtype=dtype, device=device)
        ids = idx.insert(p)
        found_ids = idx.query_points(p)
        assert sorted(found_ids) == sorted(ids), "RTree index did not return correct ids"
        pass

if __name__ == "__main__":
    # test_storage()    
    # test_rcos_index()
    # test_scor_index()
    # pass
    # test_rtree_index()
    # test_int_index()
    # pass
    test_idx_distr(idxb = partial(RTreeIndex, max_children = 10))
    pass
    # test_time(partial(test_spatial_index_range,
    #                     # index = SpatialIndex,
    #                     # index = partial(GridIndex, epsilons=1, max_bin_size = 256,
    #                     #                 switch_to_all_cap = 0.9),
    #                     # index = partial(RTreeIndex, max_children=256)
    #                   ), 
    #             store_batch_size = [2048, ], #2048, 4096], 
    #             dim_batch_size = [4, ], # 8, 16],
    #             )    
    # test_time(partial(test_spatial_index_query,
    #                     index = SpatialIndex,
    #                     # index = partial(GridIndex, epsilons=1, switch_to_all_cap = 0.9),
    #                     # index = RTreeIndex
    #                   ), 
    #             store_batch_size = [1024], #2048, 4096], 
    #             query_batch_size = [32, ], # 512, 1024],
    #             dim_batch_size = [16, 32, 64, 128, 256, 512, 1024],
    #             # dim_batch_size = [1024],
    #             # dim_batch_size = [128],
    #             # max_children = [16, 32, 64, 128, 256], # 2048, 4096],
    #             )
    
    # test_time2(test_spatial_index_query, 
    #             dict(_name = "SpatialIndex",
    #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 256),
    #             dict(_name = "GridIndex",
    #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 128,
    #                     max_children = 64),
    #             # dict(_name = "GridIndex2",
    #             #         store_batch_size = 1024, query_batch_size = 32, dim_batch_size = 64,
    #             #         max_children = 64),                        
    #             dict(_name = "RTreeIndex",
    #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 256, 
    #                     max_children = 64),
    #             # dict(_name = "RTreeIndex2",
    #             #         store_batch_size = 1024, query_batch_size = 32, dim_batch_size = 256, 
    #             #         max_children = 64),                        
    #     )

    pass 

    test_time_all(test_spatial_index_query2,
        indices = {
            "default": partial(SpatialIndex, 
                                store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 256),
            "int:1": partial(InteractionIndex, target = 0.5,
                                 store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 64),
            "rcos:1": partial(RCosIndex, target = 0.5, dtype = torch.float32,
                                 store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 64),                                 
            # "grid:1": partial(GridIndex, 
            #                     store_batch_size = 512, query_batch_size = 128, dim_batch_size = 512, max_children = 256),
            # "grid:2": partial(GridIndex, 
            #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 512, max_children = 256),
            # "grid:3": partial(GridIndex, 
            #                     store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 256),
            # "grid:4": partial(GridIndex, 
            #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 1024, max_children = 256),
            # "grid:5": partial(GridIndex, 
            #                     store_batch_size = 512, query_batch_size = 128, dim_batch_size = 512, max_children = 64),
            # "grid:6": partial(GridIndex, 
            #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 512, max_children = 64),
            "grid:7": partial(GridIndex, 
                                store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 64),
            # "grid:8": partial(GridIndex, 
            #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 1024, max_children = 64)

            "rtree:1": partial(RTreeIndex, 
                                store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 256, max_children = 64),
            "rtree:2": partial(RTreeIndex, 
                                store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 1024, max_children = 64),

            'spear:1': partial(SpearmanCorIndex, target = 0.5, dtype = torch.float32,
                                 store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 4000),
        })

    ## NOTE: best Spatial dim_batch_size=256     max_children = *
    ## NOTE: best RTree   dim_batch_size=256    max_children = 64
    ## NOTE: best Grid    dim_batch_size=128     max_children = 128/64
    # test_time(partial(test_spatial_index,
    #                     # index = SpatialIndex,
    #                     # index = partial(GridIndex, epsilons=1, max_bin_size = 64),
    #                     index = partial(RTreeIndex, max_children=35)
    #                   ), 
    #             store_batch_size = [1024, ], #2048, 4096], 
    #             dim_batch_size = [4, ], # 8, 16],
    #             )
    # plt.ioff()
    pass