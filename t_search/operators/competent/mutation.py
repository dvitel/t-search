from t_search.operators.competent.base import BaseCompetentMutation
from t_search.operators.competent.utils import DesiredSemantics, backward_desired
from t_search.syntax import Term

class CompetentMutation(BaseCompetentMutation):
    ''' Competent Mutation from Dr. Kraviec and Pawlak
        Parent program is lineary combined with random term 
    '''
    def __init__(self, 
                    **kwargs):
        super().__init__(**kwargs)
        self.term_subtree_semantics: dict[Term, dict[tuple[Term, int], tuple[DesiredSemantics, list[DesiredSemantics]]]] = {}
    
    def mutate_term(self, term: Term) -> Term | None:

        term_sem = self.evaluator.eval(term, ensure_dims=True)
        if self.semantics.is_valid(term) is False:
            self.should_break_mutation = True
            return None
        desired_term_sem = self.lib.get_desired_semantics(term, term_sem)

        if term not in self.term_subtree_semantics:
            self.desired_at_pos = backward_desired(term, self.lib.get_desired_target(), [desired_term_sem], 
                                        lambda t: self.semantics.get_outputs(t, ensure_dims=True),
                                        self.lib.get_desired_semantics, self.op_invs)
            self.term_subtree_semantics[term] = self.desired_at_pos
        else:
            self.desired_at_pos = self.term_subtree_semantics[term]
        
        child = super().mutate_term(term)

        return child