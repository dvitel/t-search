from t_search.evaluators.semantics import Semantics
from t_search.operators.competent.listener import CompetentListener
from t_search.operators.competent.mutation import CompetentMutation
from t_search.operators.competent.utils import alg_inv, backward_desired, get_desired_semantics
from t_search.operators.crossover import TermCrossover

from t_search.operators.mutation import PositionMutation
from t_search.syntax import Term, TermPos

class CompetentCrossover(TermCrossover):
    ''' Competent crossover operator '''

    def __init__(self, *, 
                    semantics: Semantics,
                    listener: CompetentListener,
                    op_invs = alg_inv,
                    max_tries: int = 2,
                    **kwargs):
        super().__init__(**kwargs)
        self.l = listener
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache
        self.max_tries = max_tries
        self.semantics = semantics

    def mutate_position(self, term: Term, position: TermPos) -> Term | None:
        child = CompetentMutation.mutate_position(self, term, position)
        return child

    def crossover_terms(self, term: Term, other_term: Term) -> Term | None:

        term_sem, other_term_sem, *_ = self.semantics.get_outputs([term, other_term], return_outputs="list")

        desired_term_sem = self.l.get_desired_semantics(term, term_sem)
        desired_other_term_sem = self.l.get_desired_semantics(other_term, other_term_sem)

        mid_point = 0.5 * (term_sem + other_term_sem)
        mid_desired = get_desired_semantics(mid_point)


        self.desired_at_pos = backward_desired(term, mid_desired, [desired_term_sem, desired_other_term_sem], 
                                     lambda t: self.semantics.get_outputs(t),
                                     self.l.get_desired_semantics, self.op_invs)
        
        child = PositionMutation.mutate_term(self, term)

        del self.desired_at_pos

        return child