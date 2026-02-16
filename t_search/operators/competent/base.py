

import torch
from t_search.base import ServiceBase
from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.semantics import Semantics
from t_search.operators.competent.utils import DesiredSemantics, get_best_constant, alg_inv, get_best_semantics, get_desired_semantics
from t_search.operators.initialization import Initialization
from t_search.operators.mutation import PositionMutation
from t_search.operators.reduction import LincombMixin
from t_search.syntax import Term
from t_search.syntax.syntax import Syntax
from t_search.syntax.term import TermPos
from t_search.utils import unique_vector_ids

class DesiredSemanticLib(ServiceBase):
    ''' Competent operators share this lib for pre-initialized set of terms.'''

    def __init__(self, *,
                    syntax: Syntax,
                    semantics: Semantics,
                    evaluator: Evaluator,
                    target: torch.Tensor, 
                    init_op: Initialization
                 ):
        self.syntax = syntax
        self.semantics = semantics
        self.evaluator = evaluator
        # self.index_init_depth = index_init_depth # if None, dynamic library - uses any available term. 
        self.term_semantics: dict[Term, DesiredSemantics] = {}        
        self.desired_target: DesiredSemantics | None = None
        self.target: torch.Tensor = target
        self.desired_target = get_desired_semantics(target)
        self.init_op = init_op #Up2D(syntax = self.syntax, depth = self.index_init_depth)
        self.lib_terms: list[Term] = []
        self.lib_vectors: torch.Tensor | None = None

    def init(self):
        lib_terms = self.init_op()
        self.evaluator.eval(lib_terms)
        valid_terms = [t for t in lib_terms if self.semantics.is_valid(t)]
        valid_vectors = self.semantics.get_outputs(valid_terms, return_type="tensor")
        unique_ids = unique_vector_ids(valid_vectors)
        self.lib_terms = [valid_terms[i] for i in unique_ids]
        self.lib_vectors = valid_vectors[unique_ids]  
        del valid_vectors  
        pass
    
    def get_desired_semantics(self, term: Term, semantics: torch.Tensor) -> DesiredSemantics:
        if term not in self.term_semantics:
            self.term_semantics[term] = get_desired_semantics(semantics)
        return self.term_semantics[term]
    
    def get_desired_target(self):
        if self.desired_target is None:
            raise ValueError("Desired target semantics not initialized.")
        return self.desired_target        


class BaseCompetentMutation(PositionMutation, LincombMixin):

    def __init__(self,
                    fitness: Fitness,   
                    semantics: Semantics,
                    evaluator: Evaluator,
                    lib: DesiredSemanticLib,
                    op_invs = alg_inv,
                    identity_atol: float = 1e-4,
                    identity_rtol: float = 1e-4,
                    with_reduction: bool = True,
                    **kwargs
                    ):
        super().__init__(**kwargs)
        self.lib = lib
        self.op_invs = op_invs
        self.fitness = fitness
        self.semantics = semantics
        self.evaluator = evaluator
        self.desired_at_pos = {} # temp cache
        self.identity_atol = identity_atol
        self.identity_rtol = identity_rtol
        self.with_reduction = with_reduction

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

        all_semantics = self.lib.lib_vectors
        
        allowed_ids, desired_vectors = get_best_semantics(desired, undesired, all_semantics)

        if len(allowed_ids) == 0:
            return None
        
        allowed_terms = [self.lib.lib_terms[i] for i in allowed_ids]
        ordered_terms = self.optimize_lincomb(allowed_terms, desired_vectors)
        best_dist, k, t, b = ordered_terms[0]
        
        new_filling = self.build_lincomb(k, t, b)

        if new_filling is None:
            return None
        
        mutated_term = self.syntax.replace_position(term, position, new_filling)

        if mutated_term is None:
            return None        
        
        trimmed_term = self.reduce_lincomb(mutated_term)

        return trimmed_term
