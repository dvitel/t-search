''' Base interface for mutation operators. '''
from math import inf
from typing import Callable, Generator, Optional, Sequence

import numpy as np

from t_search.operators.operator import Operator
from t_search.syntax.syntax import Syntax

from t_search.syntax import Term, TermPos
from t_search.syntax.flow import shuffled_position_flow

class TermMutation(Operator): 
    ''' Abstract base. Mutates population one term at a time (1-to-1 mapping pattern to repr or mutated child)'''
    def __init__(self, *, 
                 syntax: Syntax,
                 rnd: np.random.Generator, 
                 add_metrics: Callable,
                 rate : float | None = 1.0,
                 debug: bool = False):
        self.rate: float = rate
        self.cur_parents: Sequence[Term] = []
        self.rnd: np.random.Generator = rnd
        self.add_metrics = add_metrics
        self.syntax = syntax
        self.debug = debug

    def mutate_term(self, term: Term) -> Term | None:
        ''' Abstract. Mutates one term in the context of parents and already generated children ''' 
        pass # to be implemented in subclasses

    def select_terms(self, population: Sequence[Term]) -> Generator[Term, None, None]:
        ''' Produces the order of terms to try. Default: sequential order. '''
        # permuted_term_ids = self.rnd.permutation(len(population)) 
        # for term_id in permuted_term_ids:
        #     yield population[term_id]
        for term in population:
            yield term

    def __call__(self, population: Sequence[Term]) -> Sequence[Term]: 
        ''' 
            Some mutations could return None, we would like to reattempt if small number was mutated t guarantee mutated_size.
            However, we still stick to only one pass through population.
        '''

        self.cur_parents = population

        success = 0
        fail = 0     
        repr_cnt = 0    

        size = len(population)
        mutated_size = inf if self.rate is None else int(self.rate * size)
        children = [] 

        for term in self.select_terms(population):
            if mutated_size <= 0: # reproduce
                children.append(term)
                repr_cnt += 1
            else: 
                child = self.mutate_term(term)
                if child is not None:
                    success += 1
                    children.append(child)
                    mutated_size -= 1
                else:
                    fail += 1
                    children.append(term)

        self.add_metrics(success=success, fail=fail, repr=repr_cnt)        
        return children
    
class PositionMutation(TermMutation):
    ''' Abstract base. Mutates specific position inside a term. '''

    def __init__(self, *,                  
                 max_pos_tries: int = 1e6, 
                 leaf_proba: Optional[float] = 0.1, 
                 **kwargs):
        super().__init__(**kwargs)
        self.max_pos_tries = max_pos_tries
        self.leaf_proba = leaf_proba

    def select_positions(self, term: Term) -> Generator[TermPos, None, None]:   
        positions = self.syntax.get_positions(term)
        return shuffled_position_flow(positions, self.leaf_proba, self.rnd)

    def mutate_position(self, term: Term, position: TermPos) -> Term | None:
        ''' Abstract. Mutates term at the given position. '''
        pass # to be implemented in subclasses    

    def mutate_term(self, term: Term) -> Term | None | list[Term]:
        ''' Mutates one term in the context of parents and already generated children ''' 
        
        positions = self.select_positions(term)
        
        pos_try = 0
        for position in positions:
            if pos_try >= self.max_pos_tries:
                break
            pos_try += 1
            mutated_term = self.mutate_position(term, position)
            if mutated_term is not None:       
                return mutated_term
            
        return None 
