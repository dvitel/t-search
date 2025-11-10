
from typing import TYPE_CHECKING

import torch
from t_search.spatial import TermVectorStorage
from t_search.syntax import Term
from ..initialization import Up2D
from ..competent import DesiredSemantics, InversionCache, get_desired_semantics
from .base import Listener

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class CompetentListener(Listener):
    ''' Listener that is used by competent operators CM, CX
        May dynamically register programs in the library. 
        Performs initialization of the library. Injected by competent operators.
    '''

    def __init__(self, name="CL", *,
                    index: TermVectorStorage,
                    index_init_depth: int | None = None, 
                    dynamic_index: bool = False,
                    index_max_size: int = 1e10,
                 ):
        super().__init__(name)
        self.index = index # used as library of semantics 
        self.index_init_depth = index_init_depth # if None, dynamic library - uses any available term. 
        self.dynamic_index = dynamic_index
        self.index_max_size = index_max_size
        self.inv_cache = InversionCache()
        self.desired_target: DesiredSemantics | None = None

    def on_start(self, solver: 'GPSolver'):
        super().on_start(solver)
        self.desired_target = get_desired_semantics(solver.target)
        self.index.reset()
        self.inv_cache.reset()
        if self.index_init_depth is not None and self.index.len_sem() == 0: 
            init_op = Up2D(self.index_init_depth, force_pop_size=False)
            lib_terms = init_op(solver, pop_size=self.index_max_size)
            semantics = solver.eval(lib_terms, return_outputs="tensor").outputs
            self.index.insert(lib_terms, semantics) 
            del semantics

    def on_eval(self, solver, terms, semantics, fitness):
        if self.dynamic_index and self.index.len_terms() < self.index_max_size:
            self.index.insert(terms, semantics)
    
    def get_desired_semantics(self, term: Term, semantics: torch.Tensor) -> DesiredSemantics:
        if term not in self.inv_cache.term_semantics:
            self.inv_cache.term_semantics[term] = get_desired_semantics(semantics)
        return self.inv_cache.term_semantics[term]
    
    def get_desired_target(self):
        if self.desired_target is None:
            raise ValueError("Desired target semantics not initialized.")
        return self.desired_target
