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

    def __len__(self):
        return self.cur_id

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