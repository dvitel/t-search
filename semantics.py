''' Semantic storage that could potentially be indexed with spatial indexes (see spatial module)    

    It stores semantics and known mapping to Term, syntax that implement the semantics
    Unindexed implementation would scan all entries for matches if necessary.
    Also, storages allocate int indices for semantics and allow to access full vectors through getter
'''

from typing import Any

import torch

class SemanticStorage:
    ''' Abstract base class '''

    def add(self, semantics: torch.Tensor) -> int:
        ''' Add semantics to the storage and return its index. 
            semantics: torch.Tensor of shape (k,) where k is number of dimensions.
        '''
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get(self, sem_id: int) -> torch.Tensor:
        ''' Get semantics by its index.
            sem_id: int, index of the semantics in the storage.
            Returns: torch.Tensor of shape (k,) where k is number of dimensions.
        '''
        raise NotImplementedError("This method should be implemented by subclasses.")

class GlobalSemanticStorage:
    '''  '''

