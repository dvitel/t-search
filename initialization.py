''' Population initialization operators '''

from typing import Optional

import numpy as np
from spatial import VectorStorage
from term import Builder, Term, TermGenContext, gen_all_terms, grow
from scipy.spatial import ConvexHull
from typing import TYPE_CHECKING

from term_spatial import TermVectorStorage

if TYPE_CHECKING:
    from gp import GPSolver  # Import only for type checking

class Initialization:

    def __init__(self, name: str):
        self.name = name 
        self.metrics = {}

    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        return []

    def __call__(self, solver: 'GPSolver', pop_size: int):
        ''' Use to trigger initialization, pop_init should not be called directly '''
        self.metrics = {}
        population = self.pop_init(solver, pop_size)
        return population
    
class RHH(Initialization):
    ''' Ramped Half and Half initialization operator '''

    def __init__(self, name: str = "rhh", *, 
                min_depth = 1, max_depth = 5, grow_proba = 0.5,
                leaf_proba: Optional[float] = 0.1,
                freq_skew: bool = False):
        super().__init__(name)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.grow_proba = grow_proba
        self.leaf_proba = leaf_proba
        self.freq_skew = freq_skew

    def _rhh(self, solver: 'GPSolver'):
        depth = solver.rnd.randint(self.min_depth, self.max_depth + 1)
        leaf_prob = self.leaf_proba if solver.rnd.rand() < self.grow_proba else 0
        term = grow(solver.builders, grow_depth = depth, 
                    grow_leaf_prob = leaf_prob, rnd = solver.rnd, gen_metrics=self.metrics,
                    freq_skew = self.freq_skew)
        return term
    
    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        population = []
        for _ in range(pop_size):
            term = self._rhh(solver)
            # print(str(term))
            if term is not None:
                population.append(term)
        return population
    
class UpToDepth(Initialization):
    ''' All trees (without constants) up to specified size '''

    def __init__(self, name: str = "up2depth", *, depth = 2, force_pop_size: bool = False):
        super().__init__(name)
        self.depth = depth
        self.gen_context: TermGenContext | None = None
        self.force_pop_size = force_pop_size

    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        if self.gen_context is None:
            self.gen_context = TermGenContext(solver.builders.default_gen_context.min_counts,
                                            solver.builders.default_gen_context.max_counts.copy(),
                                            solver.builders.default_gen_context.arg_limits)
            self.gen_context.max_counts[solver.const_builder.id] = 0  # no constants
        population = gen_all_terms(solver.builders, depth=self.depth, start_context=self.gen_context)
        if self.force_pop_size:
            if len(population) > pop_size:
                population = solver.rnd.choice(population, size=pop_size, replace=False).tolist()
            elif len(population) < pop_size:
                pop_extend = pop_size - len(population)
                population.extend(solver.rnd.choice(population, size=pop_extend, replace=True))
        return population
    
class CachedRHH(RHH):
    ''' Considers inner terms of solver syntax cache '''
    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        if not solver.cache_terms:
            return super().pop_init(solver, pop_size)
        none_count = 0
        sz = pop_size - len(solver.vars)
        while len(solver.syntax) < sz:
            term = self._rhh(solver)
            if term is None:
                none_count += 1
            if none_count == pop_size:
                break 
        population = list(solver.syntax.values())[:sz]
        population.extend(solver.vars.values())    
        return population
    

## NOTE: semantic sampling consider only syntax
# class Ss(Initialization):
#     ''' Semantic sampling from Looks (2007). 
#         a population initialization heuristic that defines bins for programs of
#         particular sizes and fills them up to assumed capacity by semantically distinct programs
#         https://dl.acm.org/doi/pdf/10.1145/1276958.1277283
#     '''
#     def __init__(self, name: str = "Ss"):
#         super().__init__(name)

#     def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
#         return []

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
            solver.eval(leaf_terms)
            term_outputs = solver.get_cached_outputs(leaf_terms, return_tensor=True)
            self.index.insert(leaf_terms, term_outputs)
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
            solver.eval([term])
            term_output, = solver.get_cached_outputs([term], return_tensor=True)
            is_const = solver.find_any_const(term_output)
            if is_const is not None:
                continue
            self.index.insert([term], term_output.unsqueeze(0))
        population = self.index.get_repr_terms()
        return population

# class BI(Initialization):
#     ''' Behavioral initialization Jackson (2010a) '''
#     #BI invokes R HH in a loop and admits
#     # an initialized program to the population if it is semantically different from all programs
#     # already present in the population. Thus, B I is effective, but not geometric.
#     pass 

# class SGI(Initialization):
#     ''' Semantic Geometric Initialization (SGI) (Pawlak and Krawiec, 2016b) 
#         SGI surrounds a target by a set of semantics that form a convex hull that includes the target. 
#         Then, it applies domain-specific exact algorithms that explicitly construct programs for the abovementioned semantics.         
#     '''
#     def __init__(self, name: str = "SGI", *, 
#                     index: TermVectorStorage):
#         super().__init__(name)
#         self.index = index 
    
#     def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:


class CI(SDI):
    ''' Competent semantic initialization '''
    def __init__(self, name: str = "CI", *, 
                 index: TermVectorStorage):
        super().__init__(name, index=index)

    def is_point_inside_hull(self, point: np.ndarray, hull: ConvexHull, tolerance: float = 1e-12) -> bool:
        results = np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1]
        return np.all(results <= tolerance)        
    
    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        ci_population = []
        max_try_count = 3
        i = 0
        target_inside_hull = False
        target = solver.target.cpu().numpy()
        while len(ci_population) < pop_size and (i < max_try_count):
            i += 1
            population = super().pop_init(solver, i * pop_size)
            semantics = solver.get_cached_outputs(population, return_tensor=True)
            np_semantics = semantics.cpu().numpy()
            convex_hull = ConvexHull(np_semantics)
            vertex_ids = convex_hull.vertices
            ci_population = [population[vid] for vid in vertex_ids]
            target_inside_hull = self.is_point_inside_hull(target, convex_hull)
        res = ci_population
        self.metrics['target_inside_hull'] = target_inside_hull
        # res = ci_population[:pop_size]       
        return res   