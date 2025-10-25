
from typing import TYPE_CHECKING
from t_search.term import Builder, Term
from t_search.term_spatial import TermVectorStorage
from .base import Initialization

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class SDI(Initialization):
    ''' Semantically driven initialization Beadle and Johnson (2009a)
        Starts with seeding a population with single node-programs. 
        Then, it iteratively picks a random instruction and combines it with programs drawn from the population. The resulting
        program is added to the population if no other program in there has equal semantics.
    '''

    def __init__(self, name: str = "SDI", *, 
                    index: TermVectorStorage):
        super().__init__(name)
        self.index = index 
    
    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        population = self.index.get_repr_terms()
        if len(population) == 0:
            terminal_builders = solver.builders.get_leaf_builders()
            leaf_terms = [t for leaf_builder in terminal_builders for t in [leaf_builder.fn()] if t is not None]
            term_outputs = solver.eval(leaf_terms, return_outputs="tensor").outputs
            self.index.insert(leaf_terms, term_outputs)
            del term_outputs
            population = self.index.get_repr_terms()
        if len(population) >= pop_size:
            return population[:pop_size]
        
        nonterminal_builders = solver.builders.get_nonleaf_builders()
        global_try_count = 3 * (pop_size - self.index.len_sem())
        while (self.index.len_sem() < pop_size) and (global_try_count > 0): 
            global_try_count -= 1
            rnd_builder: Builder = solver.rnd.choice(nonterminal_builders)
            args = []
            try_count = 0
            for _ in range(rnd_builder.arity()):
                try_count = 1000
                while try_count > 0: 
                    arg = solver.rnd.choice(population)
                    if solver.get_depth(arg) + 1 <= solver.max_term_depth:
                        args.append(arg)
                        break
                    try_count -= 1
                if try_count == 0:
                    break 
            if try_count == 0:
                break 
            term = rnd_builder.fn(*args)
            if term is None:
                continue
            term_outputs, *_ = solver.eval(term, return_outputs="list").outputs
            is_const = solver.find_any_const(term_outputs)
            if is_const is not None:
                continue
            self.index.insert([term], term_outputs.unsqueeze(0))
        population = self.index.get_repr_terms()
        return population