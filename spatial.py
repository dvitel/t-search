''' 
Implementation of spatial indices for efficient approximate nearest neighbor search (in amortized sense).

Idea of spacial indices is to avoid full search. 
For vector x and all semantix X of size n, full search would
require O(n) comparisons of k vector values. 
'''



from dataclasses import dataclass
from itertools import product
import math
from typing import Callable, Generator, Literal, Optional, Sequence, Union
import torch
import early_exit

def get_by_ids(vectors: torch.Tensor, max_size: int, ids: None | int | tuple[int, int] | list[int]):
    if ids is None:
        return vectors[:max_size] # view
    if type(ids) is tuple:
        return vectors[ids[0]:ids[1]] # view by range
    if isinstance(ids, int):
        return vectors[ids[0]] # return a view by index 
    if len(ids) == 1:
        return vectors[ids[0]].unsqueeze(0) # also view
    return vectors[ids] # this will new tensor with values copied from mindices        

def mask_to_map(mask: torch.Tensor, key: Literal["col", "row"] = "col") -> dict[int, list[int]]:
    ''' Converts (n, m) shape mask of 0 1 to dict of 1 positions 
        key specifies should the result be the list of rows per col ("col") or cols per row ("row")
    '''
    rows_cols = torch.where(mask)
    rows = rows_cols[0].tolist() # rows in x
    cols = rows_cols[1].tolist() # cols in y
    res = {}
    if key == "col":
        for r, c in zip(rows, cols):
            res.setdefault(c, []).append(r)
        return res 
    if key == "row":
        for r, c in zip(rows, cols):
            res.setdefault(r, []).append(c) 
        return res 
    return res 

def find_close_all(x: torch.Tensor, y: torch.Tensor, rtol=1e-5, atol=1e-4) -> dict[int, list[int]]:
    ''' x y shapes (k1, n2, ... nN, dims) --> returns for each y row id, list of row ids of x which are close. '''
    R = torch.isclose(x.unsqueeze(1), y.unsqueeze(0), rtol=rtol, atol=atol).all(dim = tuple(range(2, x.ndim)))
    res = mask_to_map(R)
    return res 

def find_eq_all(x: torch.Tensor, y: torch.Tensor, rtol=1e-5, atol=1e-4) -> dict[int, list[int]]:
    ''' x y shapes (k1, n2, ... nN, dims) --> returns for each y row id, list of row ids of x which are ==. '''
    R = (x.unsqueeze(1) == y.unsqueeze(0)).all(dim = tuple(range(2, x.ndim)))
    res = mask_to_map(R)
    return res 

def find_eq(x: torch.Tensor, y: torch.Tensor, 
            id_mapping: None | tuple[int, int] | list[int] = None,
            dim_delta = 64, permute_dim_id: Callable = lambda x:x) -> list[int]:
    ''' x shape (n1, n2, ... nN, dims) 
        y shape is either (n2, ... nN, dims) --> returns (n1) 0 1 matches to y
        
    '''
    found_ids = early_exit._pred(early_exit.close_pred, x, y, 
                                 dim_delta = dim_delta, permute_dim_id = permute_dim_id)
    if isinstance(id_mapping, tuple):
        found_ids = [id_mapping[0] + i for i in found_ids]
    elif isinstance(id_mapping, list):    
        found_ids = [id_mapping[i] for i in found_ids]
    return found_ids

def find_in(tensors: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor, 
            id_mapping: None | tuple[int, int] | list[int] = None,            
            dim_delta = 64, permute_dim_id: Callable = lambda x:x) -> list[int]:
    ''' Find indices where rows are in between of tmin and tmax, tmin <= row <= tmax.
        tensors shape (N, dims), tmin and tmax shape (dims).
    '''
    # R = torch.all((tensors >= tmin) & (tensors <= tmax), dim=tuple(range(1, tensors.ndim)))
    found_ids = early_exit._pred(early_exit.range_pred, tensors, tmin, tmax,
                                    dim_delta = dim_delta, permute_dim_id = permute_dim_id)
    if isinstance(id_mapping, tuple):
        found_ids = [id_mapping[0] + i for i in found_ids]
    elif isinstance(id_mapping, list):    
        found_ids = [id_mapping[i] for i in found_ids]
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

    def __init__(self, max_size: int, dims: int, dtype = torch.float16, 
                 stats_batch_size: int = 1024, dim_delta: int = 64):
        self.capacity = max_size 
        self.dims = dims
        self.vectors = torch.empty((max_size, dims), dtype=dtype)
        self.cur_id = 0
        self.dim_delta = dim_delta
        self.stats_batch_size = stats_batch_size
        self.stats = StorageStats(self)

    def get_vectors(self, ids: None | int | list[int] | tuple[int, int]) -> torch.Tensor:
        ''' ids None --> whole storage view 
            ids int --> single vector view by id
            ids list --> tensor - copy of corresponding vectors 
        '''
        return get_by_ids(self.vectors, self.cur_id, ids)
    
    def alloc_vector(self, vector: torch.Tensor) -> int:
        ''' Adds vector to storage and returns new id. '''
        vector_id = self.cur_id
        self.cur_id += 1
        self.vectors[vector_id] = vector
        if self.cur_id - self.stats.num_vectors >= self.stats_batch_size:
            self.stats.recompute(dim_delta = self.dim_delta)
        return vector_id
    
    def _find(self, ids: None | tuple[int, int] | list[int], op, *args) -> list[int]:
        
        selection = self.get_vectors(ids)        
        permute_dim_id = lambda x:x*self.dim_delta 
        if self.stats and self.stats.var_dim_permutation:
            permute_dim_id = lambda x: self.stats.var_dim_permutation[x]
        found_ids = op(selection, *args, ids, dim_delta=self.dim_delta, 
                            permute_dim_id = permute_dim_id)
        return found_ids
    
    def find_eq(self, ids: None | tuple[int, int] | list[int], q: torch.Tensor) -> list[int]:        
        return self._find(ids, find_eq, q)
    
    def find_in(self, ids: None | tuple[int, int] | list[int], 
                tmin: torch.Tensor, tmax: torch.Tensor) -> list[int]:
        return self._find(ids, find_in, tmin, tmax)
    
    def find_mbr(self, ids: None | tuple[int, int] | list[int]) -> Optional[torch.Tensor]:
        ''' Computes min and max tensors for given ids 
            Return shape (2, dims) with min in first row and max in second row.
        '''
        if isinstance(ids, list) and len(ids) > 1: # running min max impl 
            res = torch.tensor(2, self.vectors.shape[1], dtype=self.vectors.dtype, device=self.vectors.device)
            res[0] = self.vectors[ids[0]]
            res[1] = self.vectors[ids[0]]
            for i in range(1, len(ids)):
                res[0] = torch.minimum(res[0], self.vectors[ids[i]])
                res[1] = torch.maximum(res[1], self.vectors[ids[i]])
            return res
        view = self.get_vectors(ids)
        if len(view.shape) == 1:  # single vector
            return torch.tensor([[view], [view]], dtype=view.dtype, device=view.device)
        res = torch.empty((2, view.shape[1]), dtype=view.dtype, device=view.device)
        res[0] = view.min(dim=0).values
        res[1] = view.max(dim=0).values
        return res
        
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

class SpatialIndex:
    ''' Defines the interface for spatial indices. 
        This is default implementation which isi heavy inefficient O(N), N - number of semantics.
    '''

    def __init__(self, storage: VectorStorage):
        ''' Initializes the spatial index with semantics storage 
            Indices organize storage ids instead of plain vectors.
        '''
        self.storage = storage
    
    def query_point(self, q: torch.Tensor) -> list[int]:
        ''' O(n). Return id of vector q in index if present. Empty tensor otherwise. '''
        return self.storage.find_eq(None, q) # q here is one point among N points of all_vectors off shape (N, ..., dims)
    
    def query_range(self, qrange: torch.Tensor) -> list[int]:
        ''' O(n). Returns ids stored in the index, shape (N), N >= 0 is most cases.
            qrange[0] - mins, qrange[1] - maxs, both of shape (dims).
        '''
        found_ids = self.storage.find_in(None, qrange[0], qrange[1])
        return found_ids

    def insert(self, t: torch.Tensor) -> int:
        ''' Inserts one vector t (shape (dims)) into index.
            If vector is already present - returns its vector id
            Otherwise, allocates new id in the storage.
            Default impl: O(n) as we search through all semantics
            Returns id of vector, new or old
        '''
        found_ids = self.query_point(t)
        if len(found_ids) == 0:
            return self.storage.alloc_vector(t)
        return found_ids[0]

    def query(self, q: torch.Tensor) -> list[int]:
        ''' Point and Range (rectangular) query.
            For point query, q has shape [dims], result has 0 or 1 element id depending on whether point is found.            
            For range query, q has shape [N, dims], 
                if N = 2, q is range per dimension, result is all points that are in the range.
                for N > 2, result depends on index, default behavior is to treat each ow as point and find
                           min max goting back to [2, dims] query.
        '''
        # assert 1 <= len(q.shape) <= 2, "Supporting only point and range queries with shapes (dims) or (N, dims)"
        if len(q.shape) == 1: # query point
            return self.query_point(q)
        else:
            if q.size(0) == 1:
                return self.query_point(q[0])
            if q.size(0) > 2:
                qmin = q.min(dim=0).values
                qmax = q.max(dim=0).values
                qrange = torch.stack((qmin, qmax), dim=0) # (2, dims)
            else:
                qrange = q
            return self.query_range(qrange)
                        
class GridIndex(SpatialIndex):
    ''' Grid-based spatial index for approximate NN searc.
        Splits space onto bins of fixed size. Works only with points.
        Rebuild scales down the grid to satisfy max bin size in number of points.
    '''
    
    def __init__(self, storage, epsilon: float | torch.Tensor = 1e-3, 
                max_bin_size: int = math.inf):
        ''' 
            epsilon: Size of the bin in each dimension (0 or 1 dim tensor)
            max_bin_size: Maximum number of elements in a bin, if set, 
                            grid resize (expensive) will be triggered with new epsilon that 
                            would satisfy this condition.
        '''
        super().__init__(storage)
        self.epsilon = epsilon
        self.max_bin_size = max_bin_size
        self.bins: dict[tuple, list[int]] = {} # tuple is bin index

    def _get_bin_index(self, point: torch.Tensor) -> tuple:
        ''' Get bin index for a given vector '''
        bins = point / self.epsilon  
        bin_index = tuple(bins.int().tolist())
        return bin_index
        
    def _rebuild(self, trigger_bin: list[int]):
        ''' Grid resize and bin reindex. O(n).
            Happens when max_bin_size is set and current bin size exceeds it.
        '''
        assert len(trigger_bin) > 0, "Cannot rebuild grid without bins."
        mbr = self.storage.find_mbr(trigger_bin)
        self.epsilon = (mbr[1] - mbr[0]) / 2
        del mbr
        
        old_bins = self.bins
        self.bins = {}
        for entry in old_bins.values():
            self.insert(entry) # reinsert with new epsilon
        pass

    def insert(self, t: torch.Tensor) -> int:
        ''' Add point to a grid bin. O(1), or O(s) in worst case where s - num of elements in bin. '''
        bin_index = self._get_bin_index(t)
        bin_entries = self.bins.setdefault(bin_index, [])
        if len(bin_entries) > 0:
            found_ids = self.storage.find_eq(bin_entries, t)
            if len(found_ids) > 0:
                return found_ids[0]
        new_id = self.storage.alloc_vector(t)
        bin_entries.append(t)
        if len(bin_entries) >= self.max_bin_size:
            self._rebuild(bin_entries)
        return new_id
    
    def query_point(self, q: torch.Tensor) -> list[int]:
        ''' O(1) '''
        bin_index = self._get_bin_index(q)
        bin_entries = self.bins.get(bin_index, [])
        if len(bin_entries) == 0:
            return []
        found_ids = self.storage.find_eq(bin_entries, q)
        return found_ids

    def query_range(self, qrange: torch.Tensor) -> list[int]:
        min_bin = self._get_bin_index(qrange[0])
        max_bin = self._get_bin_index(qrange[1])
        bin_ranges = product(range(b1, b2 + 1) for b1, b2 in zip(min_bin, max_bin))
        range_entries = [e for bin_index in bin_ranges for e in self.bins.get(bin_index, [])]
        if len(range_entries) == 0:
            return []
        
        found_ids = self.storage.find_in(range_entries, qrange[0], qrange[1])
        return found_ids 
        
class MBR:        

    def __init__(self, r: torch.Tensor):
        ''' r could be (dims), (K, dims) --> (2, dims) bounding box '''
        self.r = r 
        self.r_id = None
        self._area: Optional[float] = None 
        if len(r.shape) == 2:
            self.r = torch.stack((r.min(dim=0).values, r.max(dim=0).values), dim=0) # (2, dims)
            
    def is_point(self) -> bool:
        return len(self.r.shape) == 1

    def get_min(self) -> torch.Tensor:
        return self.r if self.is_point() else self.r[0]
    
    def get_max(self) -> torch.Tensor:
        return self.r if self.is_point() else self.r[1]
        
    def area(self) -> float:        
        if self._area is None:
            if self.is_point():
                self._area = 0.0
            else:
                self._area = torch.prod(self.get_max() - self.get_min()).item()
        return self._area
    
    def enlarge(self, *p: "MBR") -> "MBR":
        if len(p) == 0:
            return self
        rs = torch.stack([self.r, *(x.r for x in p)])
        new_mbr = MBR(rs)
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
    def __init__(self, storage, min_children: int = 2, max_children: int = 10, split_strategy = linear_split):
        super().__init__(storage)
        self.min_children = min_children
        self.max_children = max_children
        self.split_strategy = split_strategy
        self.root: RTreeNode | None = None

    def _insert(self, node: RTreeNode, mbr: MBR) -> tuple[list[RTreeNode], int]:
        ''' Insert point into the R-Tree node. '''
        if node.is_leaf():
            index_ids = [c.r_id for c in node.children if c.is_point() == mbr.is_point()]
            if len(index_ids) > 0:
                found_ids = self.storage.find_eq(index_ids, mbr.r) # for rerctangles, storage should return rectangles
                if len(found_ids) > 0:
                    return [], found_ids[0]
            mbr.r_id = self.storage.alloc_vector(mbr.r) #replace vector with allocated id
            node.children.append(mbr)
            replacement = []
            if len(node.children) > self.max_children:
                replacement = self.split_strategy(node, self.min_children)
            return replacement, mbr.r_id
        else:
            _, _, new_mbr, child_i = min(((c_enl, c_ar, new_mbr, i) for i, c in enumerate(node.children) for c_enl, c_ar, new_mbr in [c.enlargement(mbr)]), key = lambda x: (x[0], x[1]))
            selected_child = node.children[child_i]
            selected_child.mbr = new_mbr
            replacement, vector_id = self._insert(selected_child, mbr)
            if len(replacement) > 0: # overflow propagation
                node.children = [*node.children[:child_i], *replacement, *node.children[child_i+1:]]
                if len(node.children) > self.max_children:
                    replacement = self.split_strategy(node, self.min_children)
                else:
                    replacement = []
            return replacement, vector_id

    def insert(self, t: torch.Tensor) -> int:
        ''' Inserts one point (rects are not supported yet) ''' 
        mbr = MBR(t)
        if self.root is None:
            new_id = self.storage.alloc_vector(t)
            mbr.r_id = new_id
            self.root = RTreeNode(mbr, [mbr])
            return new_id
        else:
            replacement, vector_id = self._insert(self.root, mbr) 
            if len(replacement) > 0: # root split - need to create new root
                self.root = RTreeNode(self.root.mbr, replacement)             
            return vector_id  

    def _query(self, node: RTreeNode, mbr: MBR) -> Generator[int, None, None]:
        if node.is_leaf():
            yield from (c.vector for c in node.children if c.intersects(mbr))
        else:
            for c in node.children:
                if c.mbr.intersects(mbr):
                    yield from self._query(c, mbr)
    
    def query_point(self, q: torch.Tensor) -> list[int]:
        return self.query(q)
    
    def query_range(self, qrange: torch.Tensor) -> list[int]:
        return self.query(qrange)
    
    def query(self, q: torch.Tensor) -> list[int]:
        return list(self._query(self.root, MBR(q)))
    
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
t1 = torch.tensor([True, False, True, False], dtype=torch.bool)
res = pack_bits(t1, dtype=torch.int8)
t2 = unpack_bits(res, clamp_sz=t1.shape[-1])
assert torch.equal(t1, t2), "Packing and unpacking failed, tensors are not equal."
pass

        
class InteractionIndex(SpatialIndex):
    ''' Maps semantics to binary vector based on dynamically computed epsilons and given target. 
        One dim is one test and 0 means we far from passing the test, 1 - close.
        leaf (one interaction vector bin) splits when it has many semantics
    '''
    def __init__(self, storage: VectorStorage, target: torch.Tensor, pack_dtype = torch.uint64,
                 max_int_size: int = 64):
        super().__init__(storage)
        self.epsilons: int | torch.Tensor = 0
        sz = torch.iinfo(pack_dtype).bits
        int_dims = math.ceil(storage.dims / sz)
        self.interactions = torch.zeros((storage.capacity, int_dims), dtype=pack_dtype, device=storage.vectors.device)
        self.target = target
        self.pack_dtype = pack_dtype
        self.dims = int_dims
        self.cur_id = 0  
        self.iid_to_vids: dict[int, list[int]] = {} # interaction id to vector ids
        self.vid_to_iid: dict[int, int] = {} # vector id to interaction id
        self.max_int_size = max_int_size # for rebuild

    def get_interactions(self, ids: None | int | list[int] | tuple[int, int], decompress = False) -> torch.Tensor:
        res_tensor = get_by_ids(self.interactions, self.cur_id, ids)
        if decompress:
            res_tensor = unpack_bits(res_tensor, clamp_sz=self.dims)
        return res_tensor
    
    def get_interactions(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Should work with shapes (*n, dims), returns (*n, int_dims)'''
        distances = torch.abs(vectors - self.target)
        interactions = (distances <= self.epsilons)
        ints = pack_bits(interactions, dtype = self.pack_dtype)
        return ints
    
    def get_vector_interactions(self, vector_id: int | list[int], decompress = False) -> torch.Tensor:
        if isinstance(vector_id, list):
            int_ids = [self.vid_to_iid[v_id] for v_id in vector_id]
        elif isinstance(vector_id, int):
            int_ids = self.vid_to_iid[vector_id]
        return self.get_interactions(int_ids, decompress=decompress)

    def alloc_interactions(self, ints: torch.Tensor) -> list[int]:
        ''' Returns assigned interaction ids for a given vector ids. vec_id:int_id dict'''
        unique_interactions, unique_indices = torch.unique(ints, sorted=True, dim=0, return_inverse = True)
        if self.cur_id == 0:
            uniq_id_to_int_id = {}
        else:
            uniq_id_to_int_ids = find_eq_all(self.interactions[:self.cur_id], unique_interactions)
            uniq_id_to_int_id = {uniq_id: int_ids[0] for uniq_id, int_ids in uniq_id_to_int_ids.items()}
        not_matched_uniq_ids = [uniq_id for uniq_id in range(unique_interactions.shape[0]) if uniq_id not in uniq_id_to_int_id]
        if len(not_matched_uniq_ids) > 0:
            new_interactions = unique_interactions[not_matched_uniq_ids]
            self.interactions[self.cur_id:self.cur_id + len(not_matched_uniq_ids)] = new_interactions
            uniq_id_to_int_id.update({uniq_id: self.cur_id + i for i, uniq_id in enumerate(not_matched_uniq_ids)})
            self.cur_id += len(not_matched_uniq_ids)
        found_ids = [ uniq_id_to_int_id[uniq_id] for uniq_id in unique_indices.tolist() ]
        return found_ids
    
    def alloc_interaction(self, ints: torch.Tensor) -> int:
        ''' Returns existing or new int id '''
        found_ids = find_eq(self.interactions, ints)
        if len(found_ids) > 0:
            return found_ids[0]
        new_id = self.cur_id
        self.interactions[self.cur_id] = ints
        self.cur_id += 1
        return new_id
    
    def _rebuild(self, trigger_int_id: int): 
        ''' Happens when number of vectors for one interaction becomes greater than max_int_size. '''
        vector_ids = self.iid_to_vids[trigger_int_id]
        vectors = self.storage.get_vectors(vector_ids)
        distances = torch.abs(vectors - self.target)
        self.epsilons = distances.mean(dim=0)
        all_vectors = self.storage.get_vectors(None) # all vectors
        all_ints = self.get_interactions(all_vectors)
        self.cur_id = 0 # reset all for rebuild 
        found_ids = self.alloc_interactions(all_ints) # alloc all interactions
        self.vid_to_iid = {}
        self.iid_to_vids = {}
        for vector_id, int_id in enumerate(found_ids):
            self.iid_to_vids.setdefault(int_id, []).append(vector_id)
            self.vid_to_iid[vector_id] = int_id
        del all_ints
    
    def insert(self, t: torch.Tensor) -> int:
        ints = self.get_interactions(t)
        int_id = self.alloc_interaction(ints)
        if int_id in self.iid_to_vids:
            vector_ids = self.iid_to_vids[int_id]
            found_ids = self.storage.find_eq(vector_ids, t)
            if len(found_ids) > 0:
                return found_ids[0]
            else:
                vector_id = self.storage.alloc_vector(t)
                self.vid_to_iid[vector_id] = int_id
                vector_ids.append(vector_id)
        else:
            vector_id = self.storage.alloc_vector(t)
            self.vid_to_iid[vector_id] = int_id
            self.iid_to_vids[int_id] = [vector_id]
        if len(self.iid_to_vids[int_id]) > self.max_int_size:
            self._rebuild(int_id)
        return vector_id
        
    def query_point(self, q: torch.Tensor) -> list[int]:
        ints = self.get_interactions(q)
        int_ids = find_eq(self.interactions, ints)
        if len(int_ids) == 0:
            return []
        vector_ids = self.iid_to_vids.get(int_ids[0], [])
        if len(vector_ids) == 0:
            return []
        found_ids = self.storage.find_eq(vector_ids, q)
        return found_ids

    def query_range(self, qrange: torch.Tensor) -> list[int]:
        ints = self.get_interactions(qrange)
        imin = ints.min(dim=0).values
        imax = ints.max(dim=0).values
        int_ids = find_in(self.interactions, imin, imax)
        if len(int_ids) == 0:
            return []
        vector_ids = [vect_id for int_id in int_ids for vect_id in self.iid_to_vids[int_id]]        
        return vector_ids
    
# class RCosIndex(SpatialIndex):
#     ''' Represents torch.Tensor with only radius vector and cosine distance to target vector 
#         Splits spalce onto cones by angle and radius.
#     '''        

        
if __name__ == "__main__":
    # TODO: tests 
    # grid_index = GridIndex(epsilon=1.0, max_bin_size=5)
    # for i in range(10):
    #     grid_index.insert(IndexEntry(torch.tensor([i, i])))
    
    # print("Grid Index Query Result:", grid_index.query(torch.tensor([2.5, 2.5])))
    
    # rtree_index = RTreeIndex(min_children=2, max_children=3)
    # for i in range(10):
    #     rtree_index.insert(IndexEntry(torch.tensor([i, i])))
    
    # print("R-Tree Index Query Result:", rtree_index.query(torch.tensor([2.5, 2.5])))
    pass