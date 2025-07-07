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

@dataclass(frozen=True, eq=False, unsafe_hash=False)
class IndexEntry:
    shape: torch.Tensor 
    ''' Point or Rectangle. Some indexes support only points '''

@dataclass(frozen=True, eq=False, unsafe_hash=False)
class IndexEntryWithData(IndexEntry):
    data: Any = field(default=None)
    ''' Data associated with the point or rectangle. 
        Useful if Index is applied as dict. 
    '''    

# def get_entries_tensor(entries: Sequence[IndexEntry]) -> torch.Tensor:
#     ''' Get tensor of shapes from entries. 
#         Returns: Tensor of shape (N, dims) where N is number of entries and dims is dimension of the point.
#     '''
#     if len(entries) == 1:
#         return entries[0].shape.unsqueeze(0) # [dims] -> [1, dims]
#     return torch.stack([e.shape for e in entries], dim=0)

def get_running(entries: Iterator[torch.Tensor], running_op, init_op) -> torch.Tensor:
    first_entry = next(entries, None)
    if first_entry is None:
        return None
    res = init_op(first_entry)
    entry_id = 0 
    res = running_op(res, first_entry, entry_id)
    while entry := next(entries, None):
        entry_id += 1
        res = running_op(res, entry, entry_id)
    return res

def get_running_min(entries: Iterator[torch.Tensor]) -> torch.Tensor:
    return get_running(entries, lambda acc, x, xi: torch.minimum(acc, x, out = acc), lambda x: torch.clone(x))

def get_running_max(entries: Iterator[torch.Tensor]) -> torch.Tensor:
    return get_running(entries, lambda acc, x, xi: torch.maximum(acc, x, out = acc), lambda x: torch.clone(x))

def get_runing_argmin(entries: Iterator[torch.Tensor]) -> torch.Tensor:
    return get_running(entries, lambda acc, x, xi: torch.where(x < acc, xi, acc, out=acc), 
                                lambda x: torch.zeros_like(x, dtype=torch.int))

def get_ruuning_argmax(entries: Iterator[torch.Tensor]) -> torch.Tensor:
    return get_running(entries, lambda acc, x, xi: torch.where(x > acc, xi, acc, out=acc), 
                                lambda x: torch.zeros_like(x, dtype=torch.int))



class SpatialIndex:
    ''' Define the interface for spatial indices. '''

    # def __init__(self, sem_getter: Callable[[int], torch.Tensor]):
    #     ''' Initializes the spatial index with semantics storage 
    #         sem_getter access vector by storage id. 
    #         Indices organize storage ids instead of plain vectors.
    #     '''
    #     self.sem_getter = sem_getter

    # def rebuild(self):
    #     ''' Allows index to adjust its grouping based on current present data. 
    #         Triggered by some condition defined internally or externally.
    #         Some indices do not use this method and the perform balancing on the fly. 
    #     '''
    #     pass 

    def insert(self, entry: IndexEntry):
        ''' Inserts point into the index with associated data if any.'''
        pass 

    def query(self, q: torch.Tensor) -> Sequence[IndexEntry]:
        ''' Point and Range (rectangular) query.
            For point query, q has shape [dims], result has 0 or 1 element depending on whether point is found.            
            For range query, q has shape [N, dims], 
                if N = 2, q is range per dimension, result is all points that are in the range.
                for N > 2, result depends on index, default behavior is to trean in row as point and find
                           min max goting back to [2, dims] query.
        '''
        pass

def find_entry(entries: Iterator[torch.Tensor], t: torch.Tensor) -> int:
    ''' Find first index in entries. Returns the index or -1. '''
    for entry_id, entry in enumerate(entries):
        if torch.equal(entry, t):
            return entry_id
    return -1        

class EmptyIndex(SpatialIndex):
    ''' Always returns miss on query '''
    
    def insert(self, entry: IndexEntry):
        pass

    def query(self, q: torch.Tensor) -> Sequence[IndexEntry]:
        return []
    
class SeqIndex(SpatialIndex):
    ''' Stores entries in one big list. Very slow. '''
    def __init__(self):
        self.entries: list[IndexEntry] = []
    def insert(self, entry: IndexEntry):
        ''' Add point to the index. O(1) '''
        self.entries.append(entry)
    def query(self, q: torch.Tensor) -> Sequence[IndexEntry]:
        ''' Point query, returns first match or empty list. O(n) '''
        existing_idx = find_entry((e.shape for e in self.entries), q)
        if existing_idx >= 0:
            return [self.entries[existing_idx]]
        return []

def replace_on_collision(old_entry: Any, new_entry: Any) -> Any:
    return new_entry

class GridIndex(SpatialIndex):
    ''' Grid-based spatial index for approximate NN searc.
        Splits space onto bins of fixed size. Works only with points.
        Rebuild scales down the grid to satisfy max bin size in number of points.
    '''
    
    def __init__(self, epsilon: float | torch.Tensor = 1e-3, 
                #  epsilon_scaling: Optional[float | torch.Tensor | Callable[["GridIndex"], float | torch.Tensor]] = None
                max_bin_size: int = math.inf,
                collision: Optional[Callable[[IndexEntry, IndexEntry], IndexEntry]] = replace_on_collision):
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
        self.bins: dict[tuple, list[IndexEntry]] = {} # tuple is bin index
        self.cur_biggest_bin: list[IndexEntry] = []
        self.collision = collision

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
        
