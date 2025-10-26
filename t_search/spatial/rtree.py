from dataclasses import dataclass
from typing import Generator, Optional

import torch

from .base import SpatialIndex, VectorStorage, find_in_ranges, find_intersects, get_missing_ids, merge_ids


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