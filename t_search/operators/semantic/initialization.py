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
                 max_depth: int,
                 max_l2: float = 1e8,
                 target: torch.Tensor | None = None,
                #  num_rand_consts: int = 0,
                 size: int = 1000
                 ):
        self.syntax = syntax
        self.target = target
        self.evaluator = evaluator
        self.semantics = semantics
        self.rnd = rnd
        self.torch_gen: torch.Generator = torch_gen
        # self.num_rand_consts = num_rand_consts
        self.size = size
        self.atol = atol
        self.rtol = rtol
        self.max_depth = max_depth      
        self.max_l2 = max_l2  
        self.semantic_duplicate_retries = semantic_duplicate_retries

    def add_term(self, new_term: Term, vectors: torch.Tensor, 
                    population: list[Term], pop_under_depth: list[Term],
                    ignore_max_l2: bool = False) -> bool:
        outputs = self.evaluator.eval(new_term)
        if not self.semantics.is_valid(new_term):
            return False
        l2 = ((outputs - self.target) ** 2).sum().item()
        if not ignore_max_l2 and (l2 > self.max_l2):
            return False
        present_outputs = vectors[:len(population)]
        close_mask = torch.isclose(present_outputs, outputs, atol=self.atol, rtol=self.rtol).all(dim=1)
        if not torch.any(close_mask):
            vectors[len(population)] = outputs
            population.append(new_term)          
            if self.syntax.get_depth(new_term) < self.max_depth:
                pop_under_depth.append(new_term)
            # if not torch.any(torch.isclose(present_outputs, output, dim=1)):        
            return True
        return False
    
    def __call__(self, size: int | None = None) -> list[Term]:
        size = size if size is not None else self.size
        vectors = torch.full((size, self.semantics.dims), float("inf"), device=self.semantics.target.device, dtype=self.semantics.target.dtype)
        population = []
        pop_under_depth = []
        for var in self.syntax.get_vars():
            self.add_term(var, vectors, population, pop_under_depth,
                            ignore_max_l2=True)  
        
        for _ in range(size - len(population)):
            for _ in range(self.semantic_duplicate_retries):
                term = self.syntax.get_rand_op(lambda _: self.rnd.choice(pop_under_depth))
                if term is None:
                    continue
                was_added = self.add_term(term, vectors, population, pop_under_depth)
                if was_added:
                    break
        return population