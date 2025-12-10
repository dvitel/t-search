from t_search.evaluators.semantics import Semantics
from t_search.operators.competent.listener import CompetentListener
from t_search.operators.competent.utils import alg_inv, backward_desired, get_best_constant, get_best_semantics
from t_search.operators.mutation import PositionMutation
from t_search.syntax import Term, TermPos

class CompetentMutation(PositionMutation):
    ''' Competent Mutation from Dr. Kraviec and Pawlak
        Parent program is lineary combined with random term 
    '''
    def __init__(self, *, 
                    semantics: Semantics,
                    listener: CompetentListener,
                    op_invs = alg_inv,
                    **kwargs):
        super().__init__(**kwargs)
        self.l = listener        
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache
        self.semantics = semantics

    def mutate_position(self, term: Term, position: TermPos) -> Term | None:
        
        if (position.term, position.occur) not in self.desired_at_pos:
            return None
        
        desired, undesired = self.desired_at_pos[(position.term, position.occur)]

        all_semantics = self.l.index.get_semantics()

        best_const = get_best_constant(desired)

        if best_const is not None:
            best_term = self.syntax.get_const(value=best_const)
            mutated_term = self.syntax.replace_position(term, position, best_term)
            return mutated_term

        best_sem_id = get_best_semantics(desired, undesired, all_semantics)

        if best_sem_id is None:
            return None
        
        best_vector = all_semantics[best_sem_id]
        best_term = self.l.index.get_term_for_semantics(best_vector)
        
        mutated_term = self.syntax.replace_position(term, position, best_term)

        return mutated_term

    
    def mutate_term(self, term: Term) -> Term | None:

        term_sem = self.semantics.get_outputs(term)
        desired_term_sem = self.l.get_desired_semantics(term, term_sem)

        self.desired_at_pos = backward_desired(term, self.l.get_desired_target(), [desired_term_sem], 
                                     lambda t: self.semantics.get_outputs(t),
                                     self.l.get_desired_semantics, self.op_invs)
        
        child = super().mutate_term(term)

        del self.desired_at_pos

        return child