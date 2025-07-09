''' 
Implementation of spatial indices for efficient approximate nearest neighbor search (in amortized sense).

Idea of spacial indices is to avoid full search. 
For vector x and all semantix X of size n, full search would
require O(n) comparisons of k vector values. 
'''



from dataclasses import dataclass
from itertools import product
import math
from typing import Callable, Generator, Optional, Sequence
import torch
import early_exit

# @dataclass(frozen=True, eq=False, unsafe_hash=False)
# class IndexEntry:
#     storage_id: int

# @dataclass(frozen=True, eq=False, unsafe_hash=False)
# class IndexEntryWithData(IndexEntry):
#     data: Any = field(default=None)
#     ''' Data associated with the point or rectangle. 
#         Useful if Index is applied as dict. 
#     '''

# def get_entries_tensor(entries: Sequence[IndexEntry]) -> torch.Tensor:
#     ''' Get tensor of shapes from entries. 
#         Returns: Tensor of shape (N, dims) where N is number of entries and dims is dimension of the point.
#     '''
#     if len(entries) == 1:
#         return entries[0].shape.unsqueeze(0) # [dims] -> [1, dims]
#     return torch.stack([e.shape for e in entries], dim=0)

# def get_running(entries: Iterator[torch.Tensor], running_op, init_op) -> torch.Tensor:
#     first_entry = next(entries, None)
#     if first_entry is None:
#         return None
#     res = init_op(first_entry)
#     entry_id = 0 
#     res = running_op(res, first_entry, entry_id)
#     while entry := next(entries, None):
#         entry_id += 1
#         res = running_op(res, entry, entry_id)
#     return res

# def get_running_min(entries: Iterator[torch.Tensor]) -> torch.Tensor:
#     return get_running(entries, lambda acc, x, xi: torch.minimum(acc, x, out = acc), lambda x: torch.clone(x))

# def get_running_max(entries: Iterator[torch.Tensor]) -> torch.Tensor:
#     return get_running(entries, lambda acc, x, xi: torch.maximum(acc, x, out = acc), lambda x: torch.clone(x))

# def get_runing_argmin(entries: Iterator[torch.Tensor]) -> torch.Tensor:
#     return get_running(entries, lambda acc, x, xi: torch.where(x < acc, xi, acc, out=acc), 
#                                 lambda x: torch.zeros_like(x, dtype=torch.int))

# def get_ruuning_argmax(entries: Iterator[torch.Tensor]) -> torch.Tensor:
#     return get_running(entries, lambda acc, x, xi: torch.where(x > acc, xi, acc, out=acc), 
#                                 lambda x: torch.zeros_like(x, dtype=torch.int))

def find_eq_all(x: torch.Tensor, y: torch.Tensor, rtol=1e-5, atol=1e-4) -> dict[int, list[int]]:
    '''
        x y shapes (k1, n2, ... nN, dims) --> return (n1, k1) matrix of eq between 'rows'
    '''
    # assert len(x.shape) == len(y.shape): # 
    R = torch.isclose(x.unsqueeze(1), y.unsqueeze(0), rtol=rtol, atol=atol).all(dim = tuple(range(2, x.ndim)))
    rows_cols = torch.where(R)
    rows = rows_cols[0].tolist() # rows in x
    cols = rows_cols[1].tolist() # cols in y
    res = {}
    for r, c in zip(rows, cols):
        res.setdefault(c, []).append(r)
    return res

def find_eq(x: torch.Tensor, y: torch.Tensor, 
            id_mapping: Optional[list[int]] = None,
            permute_dim_id: Callable = lambda x:x) -> list[int]:
    ''' x shape (n1, n2, ... nN, dims) 
        y shape is either (n2, ... nN, dims) --> returns (n1) 0 1 matches to y
        
    '''
    found_ids = early_exit._pred(early_exit.close_pred, x, y, permute_dim_id = permute_dim_id)
    if id_mapping is not None:
        found_ids = [id_mapping[i] for i in found_ids]
    return found_ids

def find_in(tensors: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor, id_mapping: Optional[list[int]] = None) -> list[int]:
    ''' Find indices where rows are in between of tmin and tmax, tmin <= row <= tmax.
        tensors shape (N, dims), tmin and tmax shape (dims).
    '''
    # R = torch.all((tensors >= tmin) & (tensors <= tmax), dim=tuple(range(1, tensors.ndim)))
    found_ids = early_exit._pred(early_exit.range_pred, tensors, tmin, tmax)
    if id_mapping is not None:
        found_ids = [id_mapping[i] for i in found_ids]
    return found_ids

class StorageStats:     

    def __init__(self, storage: "VectorStorage", batch_size: int = 1024):
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
        self.batch_size = batch_size

    def recompute(self) -> None:
        delayed_count = self.storage.cur_id - self.num_vectors
        if delayed_count < self.batch_size:
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
        self.var_dim_permutation = torch.argsort(self.dim_variances, descending=True)
        if self.dim_mins is None:
            self.dim_mins = batch.min(dim=0).values
            self.dim_maxs = batch.max(dim=0).values
        else:
            self.dim_mins = torch.minimum(self.dim_mins, batch.min(dim=0).values)
            self.dim_maxs = torch.maximum(self.dim_maxs, batch.max(dim=0).values)
        del batch, mean_batch, var_batch

class VectorStorage:
    ''' Interface for storage for indexing '''

    def __init__(self, max_size: int, dims: int, dtype = torch.float16, stats_batch_size: int = 1024):
        self.vectors = torch.empty((max_size, dims), dtype=dtype)
        self.cur_id = 0
        if stats_batch_size == 0:
            self.stats = None 
        else:
            self.stats = StorageStats(self, batch_size=stats_batch_size)

    def get_vectors(self, ids: None | int | list[int] | tuple[int, int]) -> torch.Tensor:
        ''' ids None --> whole storage view 
            ids int --> single vector view by id
            ids list --> tensor - copy of corresponding vectors 
        '''
        if ids is None:
            return self.vectors[:self.cur_id] # view
        if type(ids) is tuple:
            return self.vectors[ids[0]:ids[1]] # view by range
        if isinstance(ids, int):
            return self.vectors[ids[0]] # return a view by index 
        if len(ids) == 1:
            return self.vectors[ids[0]].unsqueeze(0)
        return self.vectors[ids] # this will new tensor with values copied from mindices        
    
    def alloc_vector(self, vector: torch.Tensor) -> int:
        ''' Adds vector to storage and returns new id. '''
        vector_id = self.cur_id
        self.cur_id += 1
        self.vectors[vector_id] = vector
        if self.stats:
            self.stats.recompute()
        return vector_id
    
    # def find_eq()


# t1 = torch.tensor([
#     [0, 1, 0, 0],
#     [0, 1, 1, 0],
#     [1, 0, 0, 0],
# ])

# t [0, 1, 2, 1]



# res = torch.where(t1)
# pass

t1 = torch.tensor([ [2,3,3,4,7,6], [2,2,4,4,7,6], [1,2,3,4,5,6], [2,2,3,4,5,6], [2,3,3,4,5,6], [1,2,3,4,5,6]])
t2 = torch.tensor([[1,2,3,4,5,6], [2,2,4,4,7,6]])
t3 = torch.tensor([1,2,3,4,5,6])
t4 = torch.tensor([6,6,6,6,6,6])

R1 = torch.tensor([ 
    [   
        [2,2,3,4,7,6], 
        [2,3,4,4,7,6], 
    ],
    [
        [1,2,3,4,5,6], 
        [2,2,3,4,5,6], 
    ], 
    [
        [1,2,3,4,5,6], 
        [2,3,4,4,5,6]
    ],
    [
        [1,2,3,4,5,6], 
        [2,2,3,4,8,6], 
    ],     
    ])
R2 = torch.tensor([
    [1,2,3,4,5,6], 
    [2,3,4,4,5,6]
])
# R3 = torch.tensor([2,2,2,2,2,2])
# R4 = torch.tensor([6,6,6,6,6,6])
res = find_eq(t1, t4)

# res = find_eq(R1, R2)
res = find_in(t1, t3, t4)
pass

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
        all_vectors = self.storage.get_vectors(None)
        return find_eq(all_vectors, q) # q here is one point among N points of all_vectors off shape (N, ..., dims)
    
    def query_range(self, qmin: torch.Tensor, qmax: torch.Tensor) -> list[int]:
        ''' O(n). Returns ids stored in the index, shape (N), N >= 0 is most cases.'''
        all_vectors = self.storage.get_vectors(None)
        return find_in(all_vectors, qmin, qmax)

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
            else:
                qmin = q[0]
                qmax = q[1]
            return self.query_range(qmin, qmax)
                        
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
        self.cur_biggest_bin: list[int] = []

    def _get_bin_index(self, point: torch.Tensor) -> tuple:
        ''' Get bin index for a given vector '''
        bins = point / self.epsilon  
        bin_index = tuple(bins.int().tolist())
        return bin_index
        
    def _rebuild(self):
        ''' Grid resize and bin reindex. O(n).
            Happens when max_bin_size is set and current bin size exceeds it.
        '''
        assert len(self.cur_biggest_bin) > 0, "Cannot rebuild grid without bins."
        biggest_bin_tensors = self.storage.get_vectors(self.cur_biggest_bin)
        bin_min = biggest_bin_tensors.min(dim=0).values
        bin_max = biggest_bin_tensors.max(dim=0).values
        self.epsilon = (bin_max - bin_min) / 2
        del bin_min, bin_max
        
        old_bins = self.bins
        self.bins = {}
        self.cur_biggest_bin = []
        for entry in old_bins.values():
            self.insert(entry) # reinsert with new epsilon
        pass

    def insert(self, t: torch.Tensor) -> int:
        ''' Add point to a grid bin. O(1), or O(s) in worst case where s - num of elements in bin. '''
        bin_index = self._get_bin_index(t)
        bin_entries = self.bins.setdefault(bin_index, [])
        if len(bin_entries) > 0:
            bin_tensor = self.storage.get_vectors(bin_entries)
            found_ids = find_eq(bin_tensor, t, bin_entries)
            if len(found_ids) > 0:
                return found_ids[0]
        new_id = self.storage.alloc_vector(t)
        bin_entries.append(t)
        if len(bin_entries) > len(self.cur_biggest_bin):
            self.cur_biggest_bin = bin_entries
        if len(self.cur_biggest_bin) >= self.max_bin_size:
            self._rebuild()
        return new_id
    
    def query_point(self, q: torch.Tensor) -> list[int]:
        ''' O(1) '''
        bin_index = self._get_bin_index(q)
        bin_entries = self.bins.get(bin_index, [])
        if len(bin_entries) == 0:
            return []
        bin_tensor = self.storage.get_vectors(bin_entries)
        return find_eq(bin_tensor, q, bin_entries)

    def query_range(self, qmin: torch.Tensor, qmax: torch.Tensor) -> list[int]:
        min_bin = self._get_bin_index(qmin)
        max_bin = self._get_bin_index(qmax)
        bin_ranges = product(range(b1, b2 + 1) for b1, b2 in zip(min_bin, max_bin))
        range_entries = [e for bin_index in bin_ranges for e in self.bins.get(bin_index, [])]
        if len(range_entries) == 0:
            return []
        
        combined_tensor = self.storage.get_vectors(range_entries)
        return find_in(combined_tensor, qmin, qmax, range_entries)   
        
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
    children: list["RTreeNode" | MBR]

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
                node_tensors = self.storage.get_vectors(index_ids) # for rerctangles, storage should return rectangles
                found_ids = find_eq(node_tensors, mbr.r, index_ids)
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

    def _query(self, node: RTreeNode, mbr: MBR) -> Generator[int, None]:
        if node.is_leaf():
            yield from (c.vector for c in node.children if c.intersects(mbr))
        else:
            for c in node.children:
                if c.mbr.intersects(mbr):
                    yield from self._query(c, mbr)
    
    def query_point(self, q: torch.Tensor) -> list[int]:
        return self.query(q)
    
    def query_range(self, qmin: torch.Tensor, qmax: torch.Tensor) -> list[int]:
        return self.query(torch.stack((qmin, qmax)))
    
    def query(self, q: torch.Tensor) -> list[int]:
        return list(self._query(self.root, MBR(q)))
        
class InteractionIndex(SpatialIndex):
    ''' Maps semantics to binary vector based on dynamically computed epsilons and given target. 
        One dim is one test and 0 means we far from passing the test, 1 - close.
        leaf (one interaction vector bin) splits when it has many semantics
    '''
    def __init__(self, storage, epsilon):
        super().__init__(storage)
        self.grid_index = GridIndex(epsilon=epsilon, max_bin_size=max_bin_size)

    def insert(self, entry: IndexEntry):
        self.grid_index.insert(entry)

    def query(self, q: torch.Tensor) -> Sequence[IndexEntry]:
        return self.grid_index.query(q)
        
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