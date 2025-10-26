import math
from time import time

import torch
from .base import SpatialIndex, find_in_range, get_missing_ids, merge_ids


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