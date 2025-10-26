from operator import invert
from typing import TYPE_CHECKING

from ..mutation import PositionMutation, CM

from ..competent import InversionCache, alg_inv, get_desired_semantics
from ..initialization import Up2D
from ..base import TermsListener
from .base import TermCrossover
from syntax import Term, TermPos
from spatial import TermVectorStorage

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class CX(TermCrossover, TermsListener):
    ''' Competent crossover operator '''

    def __init__(self, name = "CX", *, 
                    index: TermVectorStorage, 
                    inv_cache: InversionCache,
                    index_init_depth: int | None = None, 
                    dynamic_index: bool = False,
                    index_max_size: int = 1e10,
                    op_invs = alg_inv,
                    max_tries: int = 2,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.index = index # used as library of semantics 
        self.inv_cache = inv_cache
        self.index_init_depth = index_init_depth # if None, dynamic library - uses any available term. 
        self.dynamic_index = dynamic_index
        self.index_max_size = index_max_size
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache
        self.max_tries = max_tries

    def op_init(self, solver):
        ''' Initializes desired combinatorial semantics and Library of programs '''
        if self.index_init_depth is not None and self.index.len_sem() == 0: 
            init_op = Up2D(self.index_init_depth, force_pop_size=False)
            lib_terms = init_op(solver, pop_size=self.index_max_size)
            semantics = solver.eval(lib_terms, return_outputs="tensor").outputs
            self.index.insert(lib_terms, semantics) 
            del semantics
        pass 

    def register_terms(self, solver, terms, semantics):
        if self.dynamic_index and self.index.len_terms() < self.index_max_size:
            self.index.insert(terms, semantics)
        return []

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        child = CM.mutate_position(self, solver, term, position)
        return child

    def crossover_terms(self, solver: 'GPSolver', term: Term, other_term: Term) -> Term | None:

        term_sem, other_term_sem, *_ = solver.eval([term, other_term], return_outputs="list").outputs

        if term not in self.term_curr:
            self.inv_cache.term_semantics[term] = get_desired_semantics(term_sem)
        if other_term not in self.term_curr:
            self.inv_cache.term_semantics[other_term] = get_desired_semantics(other_term_sem)        

        mid_point = 0.5 * (term_sem + other_term_sem)
        mid_desired = get_desired_semantics(mid_point)


        self.desired_at_pos = invert(term, mid_desired, [self.inv_cache.term_semantics[term], self.inv_cache.term_semantics[other_term]], 
                                     lambda args: solver.eval(args, return_outputs="list").outputs, 
                                     self.inv_cache.term_semantics, self.op_invs)
        
        child = PositionMutation.mutate_term(self, solver, term)

        del self.desired_at_pos

        return child