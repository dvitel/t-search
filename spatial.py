''' 
Implementation of spatial indices for efficient approximate nearest neighbor search (in amortized sense).

Idea of spacial indices is to avoid full search. 
For vector x and all semantix X of size n, full search would
require O(n) comparisons of k vector values. 
'''



from dataclasses import dataclass, field
from itertools import product
import math
from typing import Any, Callable, Generator, Iterator, Optional, Sequence
import numpy as np
import torch

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

class VectorStorage:
    ''' Interface for storage for indexing '''

    def get_vectors(self, *ids: int) -> torch.Tensor:
        ''' If called without ids, should return all currently allocated vectors 
            result shape (N, dims) - vectors of requested semantics
        '''
        pass 
    
    def alloc_vector(self, vector: torch.Tensor) -> int:
        ''' Adds vector to storage and returns new id
        '''
        pass

def find_vectors(tensors: torch.Tensor, args: list[torch.Tensor], predicate: Callable[[torch.Tensor], torch.Tensor],
                    dim_permutation: Optional[torch.Tensor] = None,
                    vectorization_threshold = 0) -> torch.Tensor:
    ''' Find indices where row matches predicate. Returns 0 1 mask.
        Predicate accepts at one step a tensor of a dimension dim_id of shape (K <= N) 
        and should output the new 0 1 mask of shape (K).
        Supports early-exit. Assumes 2d tensors (shapes (N, dims))
        Note that vectorized operation would execute many unnecessary comparisons in many cases.

        dim_permutation allows to iterate dimensions in different order
        args should be of shape (dims)
    '''
    # tensors_v = tensors.view(-1, num_dims)  # Flatten tensors to 2D, but we do not want to be generic
    num_dims = tensors.shape[-1]
    if num_dims <= vectorization_threshold:
        el_res = predicate(tensors, *args)
        res = torch.all(el_res, dim=-1)
        return res
    # mask = torch.ones(tensors.shape[0], dtype=torch.bool, device=tensors.device)    
    cur_ids = torch.arange(tensors.shape[0], device=tensors.device)
    for dim_id in (dim_permutation or range(num_dims)):
        if len(cur_ids) == 0:
            break
        cur_tensor = tensors[cur_ids, dim_id]
        dim_args = [a[dim_id] for a in args]
        mask[cur_ids] &= predicate(cur_tensor, *dim_args)
    return mask

def find_eq(tensors: torch.Tensor, t: torch.Tensor, rtol=1e-5, atol=1e-4) -> torch.Tensor:
    ''' Find indices where rows matches t.
        tensors shape (N, dims), t shape (K, dims).
    '''
    return find_vectors(tensors, [t], 
                        lambda cur_tensors, cur_t: torch.isclose(cur_tensors, cur_t, rtol=rtol, atol=atol))

def find_in(tensors: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
    ''' Find indices where rows  are in between of tmin and tmax, tmin <= row <= tmax.
        tensors shape (N, dims), tmin and tmax shape (dims).
    '''
    return find_vectors(tensors, [tmin, tmax],
                        lambda cur_tensors, cur_tmin, cur_tmax: (cur_tensors >= cur_tmin) & (cur_tensors <= cur_tmax))

t1 = torch.tensor([ [1,3,3,4,7,6], [1,2,4,4,7,6], [1,2,3,4,5,6], [2,2,3,4,5,6], [1,3,3,4,5,6], [1,2,3,4,5,6]])
t2 = torch.tensor([1,2,3,4,5,6])
res = torch.where(find_eq(t1, t2))[0]
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
    
    def query_point(self, q: torch.Tensor) -> torch.Tensor:
        ''' O(n). Return id of vector q in index if present. Empty tensor otherwise. '''
        all_vectors = self.storage.get_vectors()
        found_mask = find_eq(all_vectors, q)
        found_idxs = torch.nonzero(found_mask, as_tuple=False).squeeze()
        del found_mask
        return found_idxs
    
    def query_range(self, qmin: torch.Tensor, qmax: torch.Tensor) -> torch.Tensor:
        ''' O(n). Returns ids stored in the index, shape (N), N >= 0 is most cases.'''
        all_vectors = self.storage.get_vectors()
        found_mask = find_in(all_vectors, qmin, qmax)
        found_idxs = torch.nonzero(found_mask, as_tuple=False).squeeze()
        del found_mask
        return found_idxs

    def insert(self, t: torch.Tensor) -> torch.Tensor:
        ''' Inserts one vector t (shape (dims)) into index.
            If vector is already present - returns its vector id
            Otherwise, allocates new id in the storage.
            Default impl: O(n) as we search through all semantics
            Returns id of vector, new or old, shape (1)
        '''
        first_idx = self.query_point(t)
        if first_idx is None:
            first_idx = self.storage.alloc_vector(t) 
        return first_idx    

    def query(self, q: torch.Tensor) -> Sequence[int]:
        ''' Point and Range (rectangular) query.
            For point query, q has shape [dims], result has 0 or 1 element id depending on whether point is found.            
            For range query, q has shape [N, dims], 
                if N = 2, q is range per dimension, result is all points that are in the range.
                for N > 2, result depends on index, default behavior is to treat each ow as point and find
                           min max goting back to [2, dims] query.
        '''
        assert 1 <= len(q.shape) <= 2, "Supporting only point and range queries with shapes (dims) or (N, dims)"
        if len(q.shape) == 1: # query point
            first_idx = self.query_point(q)
            return [] if first_idx is None else [first_idx]
        else:
            if q.shape[0] == 1:
                first_idx = self.query_point(q)
                return [] if first_idx is None else [first_idx]
            if q.shape[0] > 2:
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
    
    def __init__(self, epsilon: float | torch.Tensor = 1e-3, 
                #  epsilon_scaling: Optional[float | torch.Tensor | Callable[["GridIndex"], float | torch.Tensor]] = None
                max_bin_size: int = math.inf):
        ''' 
            epsilon: Size of the bin in each dimension (0 or 1 dim tensor)
            max_bin_size: Maximum number of elements in a bin, if set, 
                            grid resize (expensive) will be triggered with new epsilon that 
                            would satisfy this condition.
        '''
            # epsilon_scaling: Scaling happens on index rebuild, by default, grid is fixed.
            #                  If callable, gets grid index and returns new epsilon.
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
        # biggest_bin = get_entries_tensor(self.cur_biggest_bin)
        bin_min = get_running_min(iter(self.cur_biggest_bin))
        bin_max = get_running_max(iter(self.cur_biggest_bin))
        self.epsilon = (bin_max - bin_min) / 2
        del bin_min, bin_max
        
        old_bins = self.bins
        self.bins = {}
        self.cur_biggest_bin = []
        for entry in old_bins.values():
            self.insert(entry) # reinsert with new epsilon
        pass

    def insert(self, entry: IndexEntry):
        ''' Add point to a grid bin. O(1), or O(s) in worst case where s - num of elements in bin. '''
        bin_index = self._get_bin_index(entry.shape)
        bin_entries = self.bins.setdefault(bin_index, [])
        if self.collision is not None:
            existing_idx = find_entry((e.shape for e in bin_entries), entry.shape)
            if existing_idx >= 0:
                bin_entries[existing_idx] = self.collision(bin_entries[existing_idx], entry)
                return
        bin_entries.append(entry)
        if len(bin_entries) > len(self.cur_biggest_bin):
            self.cur_biggest_bin = bin_entries
        if len(self.cur_biggest_bin) >= self.max_bin_size:
            self._rebuild()

    def query(self, q: torch.Tensor) -> Sequence[IndexEntry]:
        assert len(q.shape) == 1 or len(q.shape) == 2, "Query must be a point or range query."
        if len(q.shape) == 1: # point query
            bin_index = self._get_bin_index(q)
            bin_entries = self.bins.get(bin_index, [])
            existing_idx = find_entry((e.shape for e in bin_entries), q)
            if existing_idx >= 0:
                return [bin_entries[existing_idx]]  # return the found entry
            return []  # no match found
        else: # range query
            if q.size[0] == 1:
                return self.query(q[0])
            if q.size[0] > 2:
                min_max_tensor = torch.zeros(2, q.shape[1], dtype=q.dtype, device=q.device)
                min_max_tensor[0] = q.min(dim=0).values
                min_max_tensor[1] = q.max(dim=0).values
                return self.query(min_max_tensor)
            q_min = q[0]
            q_max = q[1]
            min_bin = self._get_bin_index(q_min)
            max_bin = self._get_bin_index(q_max)
            bin_ranges = product(range(b1, b2 + 1) for b1, b2 in zip(min_bin, max_bin))
            range_entries = [e for bin_index in bin_ranges for e in self.bins.get(bin_index, [])]
            if len(range_entries) == 0:
                return []
            
            # NOTE: vector operations
            # combined_bin = get_entries_tensor(range_entries)
            # matches = torch.all((q_min <= combined_bin) & (combined_bin <= q_max), dim=1)
            # match_indices = torch.nonzero(matches, as_tuple=False).squeeze().tolist()

            # NOTE: early exit
            match_indices = []
            for entry_id, bin_entry in enumerate(range_entries):
                in_range = True
                for dim_id in range(bin_entry.shape.shape[-1]):
                    if torch.any(bin_entry.shape[dim_id] < q_min[dim_id]) or torch.any(bin_entry.shape[dim_id] > q_max[dim_id]):
                        in_range = False 
                        break
                if in_range:
                    match_indices.append(entry_id)

            filetered_entries = [range_entries[i] for i in match_indices]
            return filetered_entries
        
class MBR:        

    def __init__(self, entry: IndexEntry, min_point: Optional[torch.Tensor] = None, max_point: Optional[torch.Tensor] = None):
        self.entry: IndexEntry = entry
        self._min_point: Optional[torch.Tensor] = min_point
        self._max_point: Optional[torch.Tensor] = max_point
        self._area: Optional[float] = None 

    def is_point(self):
        return len(self.entry.shape.shape) == 1
    
    def get_min_point(self):
        if self._min_point is None:
            if self.is_point():
                self._min_point = self.entry.shape
            else:
                self._min_point = self.entry.shape.min(dim=0).values
        return self._min_point
    
    def get_max_point(self):
        if self._max_point is None:
            if self.is_point():
                self._max_point = self.entry.shape
            else:
                self._max_point = self.entry.shape.max(dim=0).values
        return self._max_point
    
    def area(self) -> float:
        if self._area is None:
            self._area = torch.prod(self.get_max_point() - self.get_min_point()).item()
        return self._area
    
    def enlarge(self, *p: "MBR") -> "MBR":
        if len(p) == 0:
            return self
        if len(p) == 1:
            new_min_point = torch.minimum(self.get_min_point(), p[0].get_min_point())
            new_max_point = torch.maximum(self.get_max_point(), p[0].get_max_point())
            new_mbr = MBR(self.entry, new_min_point, new_max_point)
        else:
            # mbr_min_tensor = get_min_mbr_tensor([self, *p])
            # mbr_max_tensor = get_max_mbr_tensor([self, *p])
            new_min_point = get_running_min(self.entry.shape, *(x.entry.shape for x in p))
            new_max_point = get_running_max(self.entry.shape, *(x.entry.shape for x in p))
            new_mbr = MBR(self.entry, new_min_point, new_max_point)
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
        return torch.all(self.get_min_point() <= other.get_max_point()) and torch.all(other.get_min_point() <= self.get_max_point())

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
    if node.is_leaf():
        min_tensors = [mbr.get_min_point() for mbr in node.children]
        max_tensors = [mbr.get_max_point() for mbr in node.children]
    else:
        min_tensors = [n.mbr.get_min_point() for n in node.children]
        max_tensors = [n.mbr.get_max_point() for n in node.children]
    # L = torch.argmin(mbr_min_tensor, dim=0)  # indices of min points in each dimension
    Lt = get_runing_argmin(iter(min_tensors))
    L = Lt.tolist()
    # H = torch.argmax(mbr_max_tensor, dim=0)  # indices of max points in each dimension
    Ht = get_ruuning_argmax(iter(max_tensors))
    H = Ht.tolist()
    del L, H
    # separations = (max_tensors[H] - min_tensors[L]) / (node.mbr.get_max_point() - node.mbr.get_min_point())
    separations = [(max_tensors[h] - min_tensors[l]) / (node.mbr.get_max_point() - node.mbr.get_min_point()) for l, h in zip(L, H)]
    max_sep_dim_id = np.argmax(separations)
    selected_l_id = L[max_sep_dim_id]
    selected_h_id = H[max_sep_dim_id]
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
    def __init__(self, min_children: int = 2, max_children: int = 10, split_strategy = linear_split,
                    collision: Optional[Callable[[IndexEntry, IndexEntry], IndexEntry]] = replace_on_collision):
        self.min_children = min_children
        self.max_children = max_children
        self.split_strategy = split_strategy
        self.root: RTreeNode | None = None
        self.collision = collision
 

    def _insert(self, node: RTreeNode, mbr: MBR):
        ''' Insert point into the R-Tree node. '''
        if node.is_leaf():
            existing_idx = find_entry((c.entry.shape for c in node.children), mbr.entry.shape)
            if existing_idx >= 0: # update
                updated_entry = self.collision(node.children[existing_idx].entry, mbr.entry)
                node.children[existing_idx].entry = updated_entry
                return
            node.children.append(mbr)
            if len(node.children) > self.max_children:
                return self.split_strategy(node, self.min_children)
            return []
        else:
            _, _, new_mbr, child_i = min(((c_enl, c_ar, new_mbr, i) for i, c in enumerate(node.children) for c_enl, c_ar, new_mbr in [c.enlargement(mbr)]), key = lambda x: (x[0], x[1]))
            selected_child = node.children[child_i]
            selected_child.mbr = new_mbr
            replacement = self._insert(selected_child, mbr)
            if len(replacement) > 0: # overflow propagation
                node.children = [*node.children[:child_i], *replacement, *node.children[child_i+1:]]
                if len(node.children) > self.max_children:
                    return self.split_strategy(node, self.min_children)
            return []

    def insert(self, entry: IndexEntry):
        ''' Inserts one point or bounding rectangle of many points '''
        mbr = MBR(entry)
        if self.root is None:
            self.root = RTreeNode(mbr, [mbr])
        else:
            replacement = self._insert(self.root, mbr) 
            if len(replacement) > 0: # root split - need to create new root
                self.root = RTreeNode(self.root.mbr, replacement)                

    def _query(self, node: RTreeNode, mbr: MBR) -> Generator[IndexEntry, None]:
        if node.is_leaf():
            yield from (c.entry for c in node.children if c.intersects(mbr))
        else:
            for c in node.children:
                if c.mbr.intersects(mbr):
                    yield from self._query(c, mbr)
        
    def query(self, q: torch.Tensor) -> Sequence[IndexEntry]:
        mbr = MBR(IndexEntry(q))
        res = list(self._query(self.root, mbr))
        return res 
    
class InteractionIndex(SpatialIndex):
    ''' Maps semantics to binary vector based on dynamically computed epsilons. 
        One dim is one test and 0 means we far from passing the test, 1 - close.
        leaf (one interaction vector bin) splits when it has many semantics
    '''
    def __init__(self, epsilon: float = 1e-3, max_bin_size: int = math.inf):
        super().__init__()
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