from itertools import product
import numpy as np
from t_search.operators.initialization import Initialization
from t_search.syntax import Term
from t_search.syntax.syntax import Syntax
from t_search.syntax.term import Value

class Up2D(Initialization):
    ''' All trees (without constants) up to specified depth '''

    def __init__(self, *, 
                 syntax: Syntax,
                 rnd: np.random.Generator,
                 # from config params
                 depth = 2, 
                 const_1: bool = False,
                 max_size: int = 1000
                ):
        self.syntax = syntax
        self.depth = depth
        self.const_1 = const_1
        self.max_size = max_size
        self.rnd = rnd

    def __call__(self) -> list[Term]:
        # population = self.syntax.get_all_terms(up2depth=self.depth, max_consts=self.max_consts, const_1=self.const_1)
        depth_0 = []
        if self.const_1:
            new_const = self.syntax.one_value
            depth_0.append(new_const)
        depth_0.extend(self.syntax.get_vars())
        self.rnd.shuffle(depth_0) # we need to shuffle everything because of max_size limit
        levels = [depth_0]
        cur_size = len(depth_0)
        ops = list(self.syntax.op_builders.items())
        for d in range(1, self.depth + 1):
            if cur_size >= self.max_size:
                break
            levels_combined = [x for level in levels for x in level]            
            new_level = []
            # self.rnd.shuffle(ops)
            all_op_args = []
            for op, b in ops:
                op_arity = b.term_arity
                self.rnd.shuffle(levels_combined)
                arg_levels_ = [levels_combined] * op_arity
                if op in self.syntax.commutative:
                    arg_levels = arg_levels_.copy()
                    arg_levels[0] = levels[-1]
                    possible_args = [product(*arg_levels)]
                else:
                    possible_args = []
                    for arg_i in range(op_arity):
                        arg_levels_copy = arg_levels_.copy()
                        arg_levels_copy[arg_i] = levels[-1]
                        possible_args.append(product(*arg_levels_copy))
                all_op_args.extend([(op, args) for args_group in possible_args for args in args_group])
            self.rnd.shuffle(all_op_args)
            for op, args in all_op_args:
                new_term = self.syntax.get_op(op, *args)
                if self.syntax.is_valid(new_term) and not isinstance(new_term, Value):
                    new_level.append(new_term)
                    cur_size += 1
                    if cur_size >= self.max_size:
                        break
            levels.append(new_level)
        population = [t for level in levels for t in level]
        population.sort(key=lambda t: self.syntax._get_term_priority(t))
        return population