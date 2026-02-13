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
                 rnd: np.random.Generator,
                 torch_gen: torch.Generator,
                 atol: float = 1e-6,
                 rtol: float = 1e-5,
                 semantic_duplicate_retries: int = 1,
                #  num_rand_consts: int = 0,
                 size: int = 1000
                 ):
        self.syntax = syntax
        self.evaluator = evaluator
        self.semantics = semantics
        self.rnd = rnd
        self.torch_gen: torch.Generator = torch_gen
        # self.num_rand_consts = num_rand_consts
        self.size = size
        self.atol = atol
        self.rtol = rtol
        self.vectors = torch.full((self.size, self.semantics.dims), float("inf"), device=self.semantics.target.device, dtype=self.semantics.target.dtype)
        self.population: list[Term] = []
        self.semantic_duplicate_retries = semantic_duplicate_retries

    def add_term(self, new_term: Term):
        self.evaluator.eval(new_term)
        outputs = self.semantics.get_outputs(new_term)
        if self.semantics.is_const(outputs) is not None:
            return False
        present_outputs = self.vectors[:len(self.population)]
        close_mask = torch.isclose(present_outputs, outputs, atol=self.atol, rtol=self.rtol).all(dim=1)
        if not torch.any(close_mask):
            self.vectors[len(self.population)] = outputs
            self.population.append(new_term)             
            # if not torch.any(torch.isclose(present_outputs, output, dim=1)):        
            return True
        return False
    
    def __call__(self) -> list[Term]:
        # population = self.semantics.get_repr_terms()
        if len(self.population) > 0:
            return self.population
        # if len(population) == 0:
        for var in self.syntax.get_vars():
            self.add_term(var)
        # if self.num_rand_consts > 0:    
        #     rand_vals = torch.rand(self.num_rand_consts, generator=self.torch_gen, device=self.const_range.device, dtype=self.const_range.dtype)
        #     rand_vals = self.const_range[0] + (self.const_range[1] - self.const_range[0]) * rand_vals
        #     for const_value in rand_vals.tolist():
        #         const_term = self.syntax.get_const(value=const_value)
        #         leaf_terms.append(const_term)        
        
        for _ in range(self.size):
            for _ in range(self.semantic_duplicate_retries):
                term = self.syntax.get_rand_op(lambda _: self.rnd.choice(self.population))
                if term is None:
                    continue
                was_added = self.add_term(term)
                if was_added:
                    break
        return self.population