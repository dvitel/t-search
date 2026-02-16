from t_search.operators.competent.base import BaseCompetentMutation
from t_search.operators.competent.utils import DesiredSemantics, backward_desired, get_desired_semantics
from t_search.operators.crossover import TermCrossover

from t_search.operators.mutation import PositionMutation
from t_search.syntax import Term

class CompetentCrossover(TermCrossover, BaseCompetentMutation):
    ''' Competent crossover operator '''

    def __init__(self, 
                    **kwargs):
        super().__init__(**kwargs)
        self.term_subtree_semantics: dict[tuple[Term, Term], dict[tuple[Term, int], tuple[DesiredSemantics, list[DesiredSemantics]]]] = {}

    def select_mate(self, term: Term) -> Term | None: 
        ''' Assumes that previous operator paired best terms '''
        if self.cur_term_id % 2 == 0:
            if len(self.cur_parents) == self.cur_term_id + 1:
                self.should_break_mutation = True
                return None
            return self.cur_parents[self.cur_term_id + 1]
        else:
            return self.cur_parents[self.cur_term_id - 1]

    def crossover_terms(self, term: Term, other_term: Term) -> Term | None:

        if term == other_term:
            return None

        term_sem, other_term_sem, *_ = self.semantics.get_outputs([term, other_term], return_type="list", ensure_dims=True)

        desired_term_sem = self.lib.get_desired_semantics(term, term_sem)
        desired_other_term_sem = self.lib.get_desired_semantics(other_term, other_term_sem)

        mid_point = 0.5 * (term_sem + other_term_sem)
        mid_desired = get_desired_semantics(mid_point)

        if (term, other_term) not in self.term_subtree_semantics:
            self.desired_at_pos = backward_desired(term, mid_desired, [desired_term_sem, desired_other_term_sem], 
                                        lambda t: self.semantics.get_outputs(t, ensure_dims=True),
                                        self.lib.get_desired_semantics, self.op_invs)
            self.term_subtree_semantics[(term, other_term)] = self.desired_at_pos
        else:
            self.desired_at_pos = self.term_subtree_semantics[(term, other_term)]
        
        child = PositionMutation.mutate_term(self, term)

        return child