

from dataclasses import dataclass, field
from typing import Sequence, TYPE_CHECKING

from ..competent import DesiredSemantics, InversionCache, alg_inv, get_best_semantics, get_desired_semantics, invert

from ..initialization import Up2D
from ..base import TermsListener
from .base import PositionMutation
from syntax import Term, TermPos
from spatial import TermVectorStorage
if TYPE_CHECKING:
    from t_search.solver import GPSolver


class CM(PositionMutation, TermsListener):
    ''' Competent Mutation from Dr. Kraviec and Pawlak
        Parent program is lineary combined with random term 
    '''
    def __init__(self, name = "CM", *, 
                    index: TermVectorStorage, 
                    inv_cache: InversionCache,
                    index_init_depth: int | None = None, 
                    dynamic_index: bool = False,
                    index_max_size: int = 1e10,
                    op_invs = alg_inv,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.index = index # used as library of semantics 
        self.inv_cache = inv_cache
        self.index_init_depth = index_init_depth # if None, dynamic library - uses any available term. 
        self.dynamic_index = dynamic_index
        self.index_max_size = index_max_size
        self.desired_target: DesiredSemantics | None = None
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache

    def op_init(self, solver):
        ''' Initializes desired combinatorial semantics and Library of programs '''
        self.desired_target = get_desired_semantics(solver.target)
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
        
        if (position.term, position.occur) not in self.desired_at_pos:
            return None
        
        desired, undesired = self.desired_at_pos[(position.term, position.occur)]

        all_semantics = self.index.get_semantics()

        best_sem_id = get_best_semantics(desired, undesired, all_semantics)

        if best_sem_id is None:
            return None
        
        best_term = self.index.get_repr_term(best_sem_id)
        
        mutated_term = solver.replace_position(term, position, best_term)

        return mutated_term

    
    def mutate_term(self, solver: 'GPSolver', term: Term, parents: Sequence[Term], children: Sequence[Term]) -> Term | None:

        term_sem, *_ = solver.eval(term, return_outputs="list").outputs
        if term not in self.inv_cache.term_semantics:
            self.inv_cache.term_semantics[term] = get_desired_semantics(term_sem)

        self.desired_at_pos = invert(term, self.desired_target, [self.inv_cache.term_semantics[term]], 
                                     lambda args: solver.eval(args, return_outputs="list").outputs, 
                                     self.inv_cache.term_semantics, self.op_invs)
        
        child = super().mutate_term(solver, term, parents, children)

        del self.desired_at_pos

        return child