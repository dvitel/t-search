''' Implements opreations on semantic space 
    Indexes facilitate fast query and insertion of vector semantics 
    Spatial indexes support operation for neighborhood search
    Semantic_id - int - represents historic appearance of term during search 
    
    Possible semantic indices:
    1. Map semantic_id to semantic object (i.e. vector)
       is used for caching evaluations (avoiding reexecution of syntactic terms)
    2. Map semantic_id to terms. Use case of egraphs, where one entry represents
       equivalence class. Note, however, that we use abstract terms, and therefore these 
       indices should contain abstract leaves with bindings.
       Within equivalence class we can find smallest term to respresent the result
    3. Spatial indices, to approximate match to specific epsilon. 
'''

from typing import Any

class SemanticIndex: 

    def get(self, semantics_id: int) -> Any:
        pass

    def add(self, semantics: Any) -> int:
        ''' Add semantics to index, return semantics id '''
        pass

class DummySemanticIndex(SemanticIndex): 
    ''' Semantic_id as historic index. No tracking of duplicates '''
    def __init__(self):
        self.cur_index = 0 
        self.semantics = {}

    def get(self, semantics_id: int) -> Any:
        return self.semantics.get(semantics_id, None)
    
    def add(self, semantics: Any) -> int:
        self.semantics[self.cur_index] = semantics
        self.cur_index += 1
        return self.cur_index - 1

