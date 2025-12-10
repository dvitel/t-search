import numpy as np
import torch
from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.semantics import Semantics
from t_search.evaluators.term_spatial import TermVectorStorage
from t_search.operators.initialization import Initialization
from t_search.syntax import Term
from t_search.syntax.syntax import Syntax

class SemanticallyDrivenInitialization(Initialization):
    ''' Beadle and Johnson (2009a)
        Starts with seeding a population with single node-programs. 
        Then, it iteratively picks a random instruction and combines it with programs drawn from the population. The resulting
        program is added to the population if no other program in there has equal semantics.
    '''

    def __init__(self, *,                
                 syntax: Syntax,
                 evaluator: Evaluator,
                 semantics: Semantics,
                 const_range: torch.Tensor,
                 rnd: np.random.Generator,
                 torch_gen: torch.Generator,
                 num_rand_consts: int = 0,
                 size: int = 1000
                 ):
        self.syntax = syntax
        self.evaluator = evaluator
        self.semantics = semantics
        self.rnd = rnd
        self.torch_gen: torch.Generator = torch_gen
        self.num_rand_consts = num_rand_consts
        self.const_range = const_range
        self.size = size
    
    def __call__(self) -> list[Term]:
        population = self.semantics.get_repr_terms()
        if len(population) == 0:
            leaf_terms = list(self.syntax.get_vars())
            if self.num_rand_consts > 0:    
                rand_vals = torch.rand(self.num_rand_consts, generator=self.torch_gen, device=self.const_range.device, dtype=self.const_range.dtype)
                rand_vals = self.const_range[0] + (self.const_range[1] - self.const_range[0]) * rand_vals
                for const_value in rand_vals.tolist():
                    const_term = self.syntax.get_const(value=const_value)
                    leaf_terms.append(const_term)
            self.evaluator.eval(leaf_terms)
            population = self.semantics.get_repr_terms()
        if len(population) >= self.size:
            return population[:self.size]
        
        global_try_count = 2 * (self.size - len(population))
        while (len(population) < self.size) and (global_try_count > 0): 
            global_try_count -= 1
            term = self.syntax.get_rand_op(lambda _: self.rnd.choice(population))
            if term is None:
                continue
            term_outputs = self.evaluator.eval(term)
            const_value = self.semantics.is_const(term_outputs)
            if const_value is not None:
                continue
            population = self.semantics.get_repr_terms()
        if len(population) > self.size:
            return population[:self.size]
        return population