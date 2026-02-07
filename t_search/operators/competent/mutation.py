import torch
from t_search.evaluators.semantics import Semantics
from t_search.operators.competent.listener import CompetentListener
from t_search.operators.competent.utils import alg_inv, backward_desired, get_best_constant, get_best_semantics
from t_search.operators.mutation import PositionMutation, TermMutation
from t_search.syntax import Term, TermPos

class CompetentMutation(PositionMutation):
    ''' Competent Mutation from Dr. Kraviec and Pawlak
        Parent program is lineary combined with random term 
    '''
    def __init__(self, *, 
                    semantics: Semantics,
                    listener: CompetentListener,
                    op_invs = alg_inv,
                    syn_simplifier: TermMutation | None = None,
                    small_value: float = 1e-5,
                    **kwargs):
        super().__init__(**kwargs)
        self.l = listener        
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache
        self.semantics = semantics
        self.syn_simplifier = syn_simplifier
        self.small_value = small_value

    def mutate_position(self, term: Term, position: TermPos) -> Term | None:
        
        if (position.term, position.occur) not in self.desired_at_pos:
            return None
        
        desired, undesired = self.desired_at_pos[(position.term, position.occur)]


        best_const = get_best_constant(desired)

        if best_const is not None:
            best_term = self.syntax.get_const(value=best_const)
            # if best_term is not None:
            mutated_term = self.syntax.replace_position(term, position, best_term)
            return mutated_term

        all_semantics = self.l.index.get_semantics()
        
        best_sem_id, closest_desired, closest_test_ids = get_best_semantics(desired, undesired, all_semantics)

        if best_sem_id is None:
            return None
        
        best_vector = all_semantics[best_sem_id]
        best_term = self.l.index.get_terms_for_semantics(best_vector.unsqueeze(0))[0]
        assert best_term is not None

        # bring best_term closer to desired with linear trnsformation k * t + b

        # we compute k * ts + b that is closest to hs
        # (k * ts + b - hs)^2 --> min
        # Sx = sum(ts), Sy = sum(hs), Sxx = sum(ts^2), Sxy = sum(ts * hs)

        selected_values = best_vector[closest_test_ids]
        Sx = selected_values.sum()
        Sy = closest_desired.sum()
        Sxx = (selected_values * selected_values).sum()
        Sxy = (selected_values * closest_desired).sum()
        n = selected_values.shape[0]
        n_Covar_xy = (n * Sxy - Sx * Sy)
        n_Var_x = (n * Sxx - Sx * Sx)
        if n_Var_x / n < self.small_value: # no variation of x - x == c - searching for best b:
            b = Sy / n
            best_term = self.syntax.get_const(value=b)
        else:
            k = n_Covar_xy / n_Var_x
            b = (Sy - k * Sx) / n

            if torch.abs(k) < self.small_value:
                # approximate with constant 
                best_term = self.syntax.get_const(value=b)
            elif torch.abs(b) < self.small_value:
                # approximate with scaling only
                best_term = self.syntax.get_op("mul", self.syntax.get_const(value=k), term)
            elif torch.abs(k - 1.0) < self.small_value:
                # approximate with shifting only
                best_term = self.syntax.get_op("add", term, self.syntax.get_const(value=b))                
            else: # general case
                best_term = self.syntax.get_op("add", 
                                self.syntax.get_op("mul", self.syntax.get_const(value=k), term),
                                self.syntax.get_const(value=b)) 

        if best_term is None:
            return None
        
        mutated_term = self.syntax.replace_position(term, position, best_term)

        if self.syn_simplifier is None:
            return mutated_term
        
        new_simplified = self.syn_simplifier.mutate_term(mutated_term)

        return new_simplified

    
    def mutate_term(self, term: Term) -> Term | None:

        term_sem = self.semantics.get_outputs(term)
        desired_term_sem = self.l.get_desired_semantics(term, term_sem)

        self.desired_at_pos = backward_desired(term, self.l.get_desired_target(), [desired_term_sem], 
                                     lambda t: self.semantics.get_outputs(t),
                                     self.l.get_desired_semantics, self.op_invs)
        
        child = super().mutate_term(term)

        del self.desired_at_pos

        return child