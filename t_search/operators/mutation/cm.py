

from dataclasses import dataclass, field
from typing import Sequence, TYPE_CHECKING

from ..listeners import CompetentListener

from ..competent import alg_inv, get_best_constant, get_best_semantics, backward_desired

from .base import PositionMutation
from t_search.syntax import Term, TermPos

if TYPE_CHECKING:
    from t_search.solver import GPSolver


class CM(PositionMutation):
    ''' Competent Mutation from Dr. Kraviec and Pawlak
        Parent program is lineary combined with random term 
    '''
    def __init__(self, name = "CM", *, 
                    listener: CompetentListener,
                    op_invs = alg_inv,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.l = listener        
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        
        if (position.term, position.occur) not in self.desired_at_pos:
            return None
        
        desired, undesired = self.desired_at_pos[(position.term, position.occur)]

        all_semantics = self.l.index.get_semantics()

        best_const = get_best_constant(desired)

        if best_const is not None:
            best_term = solver.const_builder.fn(value = best_const)
            mutated_term = solver.replace_position(term, position, best_term)
            return mutated_term

        best_sem_id = get_best_semantics(desired, undesired, all_semantics)

        if best_sem_id is None:
            return None
        
        best_vector = all_semantics[best_sem_id]
        best_term = self.l.index.get_term_for_semantics(best_vector)
        
        mutated_term = solver.replace_position(term, position, best_term)

        return mutated_term

    
    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:

        term_sem, *_ = solver.eval(term, return_outputs="list").outputs
        desired_term_sem = self.l.get_desired_semantics(term, term_sem)

        self.desired_at_pos = backward_desired(term, self.l.get_desired_target(), [desired_term_sem], 
                                     lambda args: solver.eval(args, return_outputs="list").outputs, 
                                     self.l.get_desired_semantics, self.op_invs)
        
        child = super().mutate_term(solver, term)

        del self.desired_at_pos

        return child