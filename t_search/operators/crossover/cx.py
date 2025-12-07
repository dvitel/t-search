from t_search.evaluators.evaluator import Evaluator

from ..listeners import CompetentListener

from ..mutation import PositionMutation, CM

from ..competent import alg_inv, backward_desired, get_desired_semantics
from .base import TermCrossover
from t_search.syntax import Term, TermPos

class CX(TermCrossover):
    ''' Competent crossover operator '''

    def __init__(self, *, 
                    evaluator: Evaluator,
                    listener: CompetentListener,
                    op_invs = alg_inv,
                    max_tries: int = 2,
                    **kwargs):
        super().__init__(**kwargs)
        self.l = listener
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache
        self.max_tries = max_tries
        self.evaluator = evaluator

    def mutate_position(self, term: Term, position: TermPos) -> Term | None:
        child = CM.mutate_position(self, term, position)
        return child

    def crossover_terms(self, term: Term, other_term: Term) -> Term | None:

        term_sem, other_term_sem, *_ = self.evaluator.eval([term, other_term], return_outputs="list").outputs

        desired_term_sem = self.l.get_desired_semantics(term, term_sem)
        desired_other_term_sem = self.l.get_desired_semantics(other_term, other_term_sem)

        mid_point = 0.5 * (term_sem + other_term_sem)
        mid_desired = get_desired_semantics(mid_point)


        self.desired_at_pos = backward_desired(term, mid_desired, [desired_term_sem, desired_other_term_sem], 
                                     lambda args: self.evaluator.eval(args, return_outputs="list").outputs, 
                                     self.l.get_desired_semantics, self.op_invs)
        
        child = PositionMutation.mutate_term(self, term)

        del self.desired_at_pos

        return child