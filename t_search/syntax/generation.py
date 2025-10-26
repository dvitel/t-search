from collections import deque
from dataclasses import dataclass
import inspect
import math
from typing import Callable, Optional
import numpy as np

from .term import Term


def alloc_tape(width: int, penalties: list[tuple[list[int] | int, float, float]] = [],
                buf_n:int = 100, rnd: np.random.RandomState = np.random,
                freq_skew: np.ndarray | None = None) -> np.ndarray:
    weights = rnd.random((buf_n, width))
    if freq_skew is not None:
        weights *= freq_skew
    for ids, p, level in penalties:
        selection = weights[:,ids]
        weights[:,ids] = np.where(selection >= p, level, 0)
    return weights # smaller is better

def check_tape(pos_id: int, tape, 
                    penalties: list[tuple[list[int] | int, float]] = [],
                    buf_n:int = 100, rnd: np.random.RandomState = np.random,
                    freq_skew: np.ndarray | None = None) -> np.ndarray:    
    if pos_id >= tape.shape[0]:
        new_tape = np.zeros((tape.shape[0] + buf_n, tape.shape[1]), dtype=tape.dtype)
        new_tape[:tape.shape[0]] = tape
        new_part = alloc_tape(tape.shape[1], penalties=penalties, buf_n=buf_n, 
                              rnd=rnd, freq_skew = freq_skew)
        new_tape[new_tape.shape[0] - buf_n:] = new_part
        tape = new_tape
    return tape

def _add_factorize(total: int, min_counts: np.ndarray, max_counts: np.ndarray, 
                    rnd: np.random.RandomState = np.random) -> np.ndarray | None:
    ''' Splits total onto additives: total = sum(res) s.t. res under count constraints'''

    # permutation = rnd.permutation(len(min_counts))

    # min_counts = min_counts[permutation]
    # max_counts = max_counts[permutation]

    # total_mins = np.array([0, *np.cumsum(min_counts)])
    # total_maxs = np.array([0, *np.cumsum(max_counts)])
    total_mins = np.sum(min_counts)
    total_maxs = np.sum(max_counts)

    res = [0 for _ in range(len(min_counts))] 

    for i in rnd.permutation(len(min_counts)):
        cur_min = min_counts[i]
        cur_max = max_counts[i]
        total_mins -= cur_min
        total_maxs -= cur_max
        real_min = max(cur_min, total - total_maxs)
        real_max = min(cur_max, total - total_mins)
        if real_min > real_max:
            return None
        new_count = rnd.randint(real_min, real_max + 1)
        total -= new_count
        res[i] = new_count

    counts = np.array(res, dtype=int)

    return counts

# test3 = _add_factorize(1, np.array([1,1,1]), np.array([5, 5, 5]))
# test3 = _add_factorize(3, np.array([0, 1]), np.array([5, 10]))
# test1 = _add_factorize(10, np.array([1, 1, 0]), np.array([3, 3, 1]))
# test2 = _add_factorize(5, np.array([2, 1, 3]), np.array([3, 3, 3]))
# test4 = _add_factorize(5, np.array([0, 0, 0]), np.array([3, 3, 3]))
# pass 

def get_fn_arity(fn: Callable) -> int:
    signature = inspect.signature(fn)
    params = [p for p in signature.parameters.values() if p.kind != inspect.Parameter.KEYWORD_ONLY]
    return len(params)

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class Builder:
    name: str
    fn: Callable
    term_arity: int
    min_count: int | None = None
    max_count: int | None = None
    context_limits: np.ndarray | None = None
    ''' Specifies maximum number of builder occurances under this builder,
        int 1d array of size (num_builders,), combines with max_counts eventually 
    '''
    arg_limits: np.ndarray | None = None
    ''' For each argumemt specifies allowed builders 0/1 bool mask of size (arity, num_builders) '''
    # commutative: bool = False

    def __post_init__(self):
        self.id: int | None = None
        self.leaf_id_id: int | None = None
        self.nonleaf_id_id: int | None = None 
        if self.term_arity == 1:
            self.commutative = True 

    def arity(self) -> int:
        return self.term_arity

class Builders:

    def __init__(self, builders: list[Builder], get_term_builder: Callable[[Term], Builder],
                    disallow_initial_leaves: bool = True, max_depth = 17,
                    global_min_count: np.int8 = 0, global_max_count: np.int8 = 100):
        self.builders: list[Builder] = builders
        self.get_term_builder: Callable[[Term], Builder] = get_term_builder
        self.leaf_ids = []
        self.nonleaf_ids = []
        self.global_max_count = global_max_count
        for bi, b in enumerate(self.builders):
            b.id = bi
            if b.arity() == 0:
                b.leaf_id_id = len(self.leaf_ids)
                self.leaf_ids.append(bi)
            else:
                b.nonleaf_id_id = len(self.nonleaf_ids)
                self.nonleaf_ids.append(bi)
            if b.max_count is None:
                b.max_count = global_max_count
            if b.min_count is None:
                b.min_count = global_min_count
        self.leaf_ids = np.array(self.leaf_ids, dtype=np.int8)
        self.nonleaf_ids = np.array(self.nonleaf_ids, dtype=np.int8)
        self.min_counts: np.ndarray = np.array([b.min_count for b in self.builders], dtype=np.int8)
        self.max_counts: np.ndarray = np.array([b.max_count for b in self.builders], dtype=np.int8)
        self.arity_builder_ids: dict[int, np.ndarray] = {}
        self.max_arity = 0
        for bi, b in enumerate(self.builders):
            self.arity_builder_ids.setdefault(b.arity(), []).append(bi)
            self.max_arity = max(self.max_arity, b.arity())

        self.arity_builder_ids = {a: np.array(self.arity_builder_ids[a], dtype=np.int8) for a in sorted(self.arity_builder_ids.keys())}
        
        self.zero = np.zeros((len(self.builders),), dtype=np.int8)
        self.unlimited = np.full((len(self.builders),), self.global_max_count, dtype=np.int8)
        self.one_hot = np.eye(len(self.builders), dtype=np.int8)

        initial_arg_limits = None 
        if disallow_initial_leaves:
            initial_arg_limits = self.unlimited.copy()
            initial_arg_limits[self.leaf_ids] = 0

        self.has_leaf_min_counts=np.any(self.min_counts[self.leaf_ids] > 0)
        self.has_nonleaf_min_counts=np.any(self.min_counts[self.nonleaf_ids] > 0)

        self.default_gen_context = TermGenContext(
            min_counts=self.min_counts,
            max_counts=self.max_counts,
            arg_limits=initial_arg_limits)
        
        self.max_depth = max_depth
        self.max_leaf_count_per_depth = [1] 
        for _ in range(1, self.max_depth + 1):
            new_count = self.max_leaf_count_per_depth[-1] * self.max_arity
            self.max_leaf_count_per_depth.append(new_count)

        
    def __len__(self):
        return len(self.builders)
    
    def get_term_counts(self, max_depth: int) -> np.ndarray: 
        ''' Returns number of terms that start with corresponding builder and then have arbitrary structure 
            NOTE: this is nonprecise estimation as it does not take into account constraints 
        '''
        res = np.zeros((len(self.builders),), dtype=np.int64)
        res[self.leaf_ids] = 1
        for d in range(1, max_depth + 1):
            new_res = np.zeros((len(self.builders),), dtype=np.int64)
            new_res[self.leaf_ids] = 1
            prev_total = np.sum(res)
            new_res[self.nonleaf_ids] = [prev_total ** self.builders[bi].arity() for bi in self.nonleaf_ids]
            res = new_res
        return res

    def get_leaf_builders(self) -> list[Builder]:
        return [self.builders[bi] for bi in self.leaf_ids]

    def get_nonleaf_builders(self) -> list[Builder]:
        return [self.builders[bi] for bi in self.nonleaf_ids]
    
    def limit_context(self, cl: dict[Builder, dict[Builder, int]]) -> 'Builders':
        for builder, limits in cl.items():            
            builder.context_limits = self.unlimited.copy()
            for bi, b in enumerate(self.builders):
                if b in limits:
                    builder.context_limits[bi] = limits[b]

    def limit_args(self, al: dict[Builder, dict[Builder, int]]) -> 'Builders':
        for builder, limits in al.items():            
            builder.arg_limits = self.unlimited.copy()
            for bi, b in enumerate(self.builders):
                if b in limits:
                    builder.arg_limits[bi] = limits[b]
        
@dataclass(frozen=False, eq=False, unsafe_hash=False)    
class TermGenContext:
    ''' When we generate term, we preserve point requirements for later poitn regeneration '''

    min_counts: np.ndarray
    max_counts: np.ndarray
    arg_limits: np.ndarray | None = None

    def can_alloc(self, op_id: int, counts: np.ndarray, arg_counts: np.ndarray) -> bool:
        return (counts[op_id] < self.max_counts[op_id]) and \
                ((self.arg_limits is None) or (arg_counts[op_id] < self.arg_limits[op_id]))
                # (counts[op_id] < self.context_limits[op_id])
    
    def split(self, term_id: int, term_counts: np.ndarray, term_left_args: int, leaf_ids: np.ndarray,
                term_context_limits: np.ndarray | None = None, term_arg_limits: np.ndarray | None = None,
                rnd: np.random.RandomState = np.random) -> 'TermGenContext':
        
        left_min_counts = self.min_counts - term_counts # term counts - num of nodes in term including root
        left_max_counts = self.max_counts - term_counts
        if term_context_limits is not None:
            term_context_limits_with_root = term_context_limits.copy()
            term_context_limits_with_root[term_id] += 1
            arg_context_limits = term_context_limits_with_root - term_counts
            left_max_counts = np.minimum(left_max_counts, arg_context_limits)
            
        left_min_counts[left_min_counts < 0] = 0
        if term_left_args == 1:
            arg_min_counts = left_min_counts
            arg_max_counts = left_max_counts
        else:
            max_of_min_counts = left_min_counts // term_left_args
            arg_min_counts = rnd.randint(0, max_of_min_counts + 1)

            left_leaf_max_counts = left_max_counts[leaf_ids]
            for _ in range(term_left_args - 1):
                allowed_leaf_id_ids, = np.where(left_leaf_max_counts > 0)
                selected_leaf_id_id = rnd.choice(allowed_leaf_id_ids)
                leaf_id = leaf_ids[selected_leaf_id_id]
                left_max_counts[leaf_id] -= 1
                left_leaf_max_counts[selected_leaf_id_id] -= 1
            pass


            # max_of_max_counts = left_max_counts // term_left_args
            # arg_max_counts = rnd.randint(arg_min_counts, max_of_max_counts + 1)
            arg_max_counts = left_max_counts # // term_left_args

        return TermGenContext(arg_min_counts, arg_max_counts, term_arg_limits)

    def max_split(self, term_id: int, term_counts: np.ndarray, term_left_args: int,
                term_context_limits: np.ndarray | None = None, term_arg_limits: np.ndarray | None = None) -> 'TermGenContext':
        
        left_min_counts = self.min_counts - term_counts # term counts - num of nodes in term including root
        left_max_counts = self.max_counts - term_counts
        if term_context_limits is not None:
            term_context_limits_with_root = term_context_limits.copy()
            term_context_limits_with_root[term_id] += 1
            arg_context_limits = term_context_limits_with_root - term_counts
            left_max_counts = np.minimum(left_max_counts, arg_context_limits)
            
        left_min_counts[left_min_counts < 0] = 0
        if term_left_args == 1:
            arg_min_counts = left_min_counts
            arg_max_counts = left_max_counts
        else:
            arg_min_counts = np.zeros_like(left_min_counts)

            arg_max_counts = left_max_counts # // term_left_args

        return TermGenContext(arg_min_counts, arg_max_counts, term_arg_limits)

global_gen_id = 0 # for debugging

def gen_term(builders: Builders, 
            max_depth = 5, leaf_proba: float | None = 0.1,
            rnd: np.random.RandomState = np.random, buf_n = 100, inf = 100,
            start_context: TermGenContext | None = None,
            arg_counts: np.ndarray | None = None,
            gen_metrics: dict | None = None,
            freq_skew: bool = False
         ) -> Optional[Term]:
    ''' Arities should be unique and provided in sorted order.
        Counts should correspond to arities 
    '''
    global global_gen_id
    global_gen_id += 1

    # metrics 
    backtracks = 0 
    gen_fails = 0

    penalties = [] if leaf_proba is None else [(builders.leaf_ids, leaf_proba, 1)]

    if freq_skew:
        term_counts = builders.get_term_counts(max_depth)
        total_counts = np.sum(term_counts)
        freq_skew = term_counts / total_counts
    else:
        freq_skew = None

    tape = alloc_tape(len(builders), penalties=penalties, buf_n=buf_n, rnd=rnd, freq_skew = freq_skew) # tape is 2d ndarray: (t, score)

    pos_id = 0 
    def get_next_tape_values():
        nonlocal tape, pos_id 
        tape = check_tape(pos_id, tape, penalties=penalties, buf_n=buf_n, rnd=rnd, freq_skew = freq_skew)
        tape_values = tape[pos_id]
        pos_id += 1
        return tape_values
        
    def _gen_rec(
        gen_context: TermGenContext,
        counts: np.ndarray,
        arg_counts: np.ndarray,
        at_depth: int) -> Optional[Term]:
        nonlocal backtracks
        
        leaf_min_count = 0 if builders.has_leaf_min_counts else gen_context.min_counts[builders.leaf_ids].sum()
        nonleaf_min_count = 0 if builders.has_nonleaf_min_counts else gen_context.min_counts[builders.nonleaf_ids].sum()

        
        if at_depth == max_depth: # leaf forced

            if nonleaf_min_count > 0 or leaf_min_count > 1:
                # allocating leaf will not sat min requirements
                return None 
            
            if leaf_min_count == 1: # exactly one leaf is required
                op_id_ids, = np.where(gen_context.min_counts[builders.leaf_ids] == 1)
                op_id = builders.leaf_ids[op_id_ids[0]]

                if not gen_context.can_alloc(op_id, counts, arg_counts):
                    return None

                new_term = builders.builders[op_id].fn()
                if new_term is not None: # on success we dec all requirements to tighten the following generations
                    counts[op_id] += 1
                    arg_counts[op_id] += 1

                return new_term 
            
            else: # at depth, leaf, no min requirements
                            
                tape_values = get_next_tape_values() # rand values
                tape_values[builders.nonleaf_ids] = inf 

                while True: # trying different leaves

                    op_id = np.argmin(tape_values)
                    cur_val = tape_values[op_id]
                    if cur_val >= inf:
                        break
                    if not gen_context.can_alloc(op_id, counts, arg_counts):
                        tape_values[op_id] = inf
                        continue
                    
                    new_term = builders.builders[op_id].fn()
                    if new_term is not None: 
                        counts[op_id] += 1
                        arg_counts[op_id] += 1

                    return new_term

                return None 
        
        # not at max depth, non-leaf possible

        # we estimate minimal arity to filter out non-leafs that would not satisfy
        min_arity = math.ceil(leaf_min_count / builders.max_leaf_count_per_depth[max_depth - at_depth - 1])

        max_leaf_count = gen_context.max_counts[builders.leaf_ids].sum()        

        # NOTE: for future, we also can constrain min_arity by maximal possible non-leaves in the tree
        # min_arity = math.ceil(nonleaf_min_count / max_nonleaf_count_per_depth[max_depth - at_depth - 1])

        # max arity - assuming instant leaves, op arity cannot be greater than leaf max allowed count
        
        # arity cannot be higher of max requirements 

        # for arity in range(builders.max_arity):
        #     if arity < min_arity:
        #         tape_values[builders.arity_builder_ids[arity]] = inf
            
        tape_values = get_next_tape_values()

        if nonleaf_min_count > 0 or leaf_min_count > 1:
            tape_values[builders.leaf_ids] = inf

        while True:
            op_id = np.argmin(tape_values)
            cur_val = tape_values[op_id]
            if cur_val >= inf: # no more valid ops
                break

            builder = builders.builders[op_id]
            op_arity = builder.arity()

            if not gen_context.can_alloc(op_id, counts, arg_counts):
                tape_values[op_id] = inf
                continue

            if op_arity == 0: # leaf selected
                
                if leaf_min_count == 1: # exactly one leaf is required
                    op_id_ids, = np.where(gen_context.min_counts[builders.leaf_ids] == 1)
                    op_id = builders.leaf_ids[op_id_ids[0]]

                    new_term = builders.builders[op_id].fn()
                    if new_term is not None: # on success we dec all requirements to tighten the following generations
                        counts[op_id] += 1
                        arg_counts[op_id] += 1

                    return new_term 
                
                else: # no min requirements
                                                        
                    new_term = builders.builders[op_id].fn()
                    if new_term is not None: 
                        counts[op_id] += 1
                        arg_counts[op_id] += 1
                        return new_term
                    else:
                        tape_values[op_id] = inf
                        continue
            
            # non-leaf selected, we estimate if min leaf requirements could be satisfied with op arity, max arity, depth and given count
        
            if op_arity < min_arity: # we cannot satisfy min leaf count with this arity
                tape_values[op_id] = inf 
                continue

            if op_arity > max_leaf_count:
                assert max_leaf_count > 0
                tape_values[op_id] = inf 
                continue
            
            new_counts = np.zeros_like(counts)
            new_arg_counts = np.zeros_like(counts)
            new_counts[op_id] += 1
            arg_ops = []
            # print(f"\t{builder.name}? {at_depth} {gen_context.min_counts}:{gen_context.max_counts}")
            backtrack = False
            for arg_i in range(op_arity):

                arg_gen_context = gen_context.split(op_id, new_counts, op_arity - arg_i, builders.leaf_ids, builder.context_limits, builder.arg_limits, rnd=rnd)
            
                arg_term = _gen_rec(arg_gen_context, new_counts, new_arg_counts, at_depth + 1)
                if arg_term is not None:
                    arg_ops.append(arg_term)
                else:
                    # print(f"\t<<< {at_depth} {arg_i_min_counts}:{arg_i_max_counts}")
                    backtrack = True
                    break
            if backtrack:
                tape_values[op_id] = inf
                backtracks += 1
                continue
            new_term = builder.fn(*arg_ops)
            if new_term is None:
                tape_values[op_id] = inf
                continue

            # real_term_builder = builders.get_term_builder(new_term)
            # real_term_builder.id

            counts += new_counts
            arg_counts[op_id] += 1

            assert np.all(new_counts >= gen_context.min_counts), f"Min counts violation: {new_counts} < {gen_context.min_counts}"
            assert np.all(new_counts <= gen_context.max_counts), f"Max counts violation: {new_counts} > {gen_context.max_counts}"
            assert (gen_context.arg_limits is None) or np.all(arg_counts <= gen_context.arg_limits), f"Args counts violaton: {arg_counts} > {builder.arg_limits}"
            # print(str(new_term))
            return new_term
        return None

    if start_context is None:
        start_context = builders.default_gen_context

    counts = builders.zero.copy()
    if arg_counts is None:
        arg_counts = counts.copy()

    new_term = _gen_rec(start_context, counts, arg_counts, 0)

    if new_term is None:
        print(f"Fail generate {global_gen_id}: \n{str(start_context)}\nreason={new_term}")
        gen_fails += 1
        return None

    assert np.all(counts >= start_context.min_counts)
    assert np.all(counts <= start_context.max_counts)

    if gen_metrics is not None:
        gen_metrics['backtracks'] = gen_metrics.get('backtracks', 0) + backtracks
        gen_metrics['gen_fails'] = gen_metrics.get('gen_fails', 0) + gen_fails

    return new_term

def gen_all_terms(builders: Builders, depth = 3,
            start_context: TermGenContext | None = None,
            arg_counts: np.ndarray | None = None
         ) -> list[Term]:
    ''' Generate all terms up to given depth under given constraints. '''
        
    def _gen_all_rec(
        gen_context: TermGenContext,
        counts: np.ndarray,
        arg_counts: np.ndarray,
        at_depth: int) -> list[tuple[Term, np.ndarray, np.ndarray]]:
        
        leaf_min_count = 0 if builders.has_leaf_min_counts else gen_context.min_counts[builders.leaf_ids].sum()
        nonleaf_min_count = 0 if builders.has_nonleaf_min_counts else gen_context.min_counts[builders.nonleaf_ids].sum()
        
        if at_depth == depth: # leaf forced

            if nonleaf_min_count > 0 or leaf_min_count > 1:
                # allocating leaf will not sat min requirements
                return [] 
            
            if leaf_min_count == 1: # exactly one leaf is required
                op_id_ids, = np.where(gen_context.min_counts[builders.leaf_ids] == 1)
                op_id = builders.leaf_ids[op_id_ids[0]]

                if not gen_context.can_alloc(op_id, counts, arg_counts):
                    return []

                new_term = builders.builders[op_id].fn()
                if new_term is not None: # on success we dec all requirements to tighten the following generations
                    counts[op_id] += 1
                    arg_counts[op_id] += 1
                    return [(new_term, counts, arg_counts)] 
                
                return []
            
            else: # at depth, leaf, no min requirements
                            
                new_term_ops = []

                for op_id in builders.leaf_ids:
                    if not gen_context.can_alloc(op_id, counts, arg_counts):
                        continue
                
                    new_term = builders.builders[op_id].fn()
                    if new_term is not None: 
                        new_term_ops.append((new_term, op_id))

                new_terms = []
                for i, (new_term, op_id) in enumerate(new_term_ops):
                    if i < len(new_term_ops) - 1:
                        new_counts = counts.copy()
                        new_arg_counts = arg_counts.copy()
                        new_counts[op_id] += 1
                        new_arg_counts[op_id] += 1
                        new_terms.append((new_term, new_counts, new_arg_counts))
                    else: # last term, no need to copy
                        counts[op_id] += 1
                        arg_counts[op_id] += 1
                        new_terms.append((new_term, counts, arg_counts))

                return new_terms
        
        # not at max depth, non-leaf possible

        # we estimate minimal arity to filter out non-leafs that would not satisfy
        min_arity = math.ceil(leaf_min_count / builders.max_leaf_count_per_depth[depth - at_depth - 1])

        max_leaf_count = gen_context.max_counts[builders.leaf_ids].sum()        

        # NOTE: for future, we also can constrain min_arity by maximal possible non-leaves in the tree
        # min_arity = math.ceil(nonleaf_min_count / max_nonleaf_count_per_depth[max_depth - at_depth - 1])

        # max arity - assuming instant leaves, op arity cannot be greater than leaf max allowed count
        
        # arity cannot be higher of max requirements 

        # for arity in range(builders.max_arity):
        #     if arity < min_arity:
        #         tape_values[builders.arity_builder_ids[arity]] = inf
            
        op_ids = [*builders.nonleaf_ids]

        if not(nonleaf_min_count > 0 or leaf_min_count > 1):
            if leaf_min_count == 1:
                op_id_ids, = np.where(gen_context.min_counts[builders.leaf_ids] == 1)
                op_id = builders.leaf_ids[op_id_ids[0]]
                op_ids.append(op_id)
            else:
                op_ids.extend(builders.leaf_ids)

        new_terms = []

        for op_i, op_id in enumerate(op_ids):

            builder = builders.builders[op_id]
            op_arity = builder.arity()

            if not gen_context.can_alloc(op_id, counts, arg_counts):
                continue

            if op_arity == 0: # leaf selected                
                                                        
                new_term = builders.builders[op_id].fn()
                if new_term is not None: 
                    if op_i == len(op_ids) - 1: # last term, no need to copy
                        counts[op_id] += 1
                        arg_counts[op_id] += 1
                        new_terms.append((new_term, counts, arg_counts))
                    else:
                        new_counts = counts.copy()
                        new_arg_counts = arg_counts.copy()
                        new_counts[op_id] += 1
                        new_arg_counts[op_id] += 1
                        new_terms.append((new_term, new_counts, new_arg_counts))
                continue
            
            # non-leaf selected, we estimate if min leaf requirements could be satisfied with op arity, max arity, depth and given count
        
            if op_arity < min_arity: # we cannot satisfy min leaf count with this arity
                continue

            if op_arity > max_leaf_count:
                continue
            
            new_counts = np.zeros_like(counts)
            new_arg_counts = np.zeros_like(counts)
            new_counts[op_id] += 1
            # arg_ops = []
            arg_q = deque([([], new_counts, new_arg_counts)])
            while len(arg_q) > 0:
                cur_args, cur_counts, cur_arg_counts = arg_q.popleft()
                if len(cur_args) == op_arity:
                    new_term = builder.fn(*cur_args)
                    if new_term is not None:
                        all_counts = counts + cur_counts
                        all_arg_counts = arg_counts.copy()
                        all_arg_counts[op_id] += 1
                        new_terms.append((new_term, all_counts, all_arg_counts))
                else: # need new arg
                    arg_gen_context = gen_context.max_split(op_id, cur_counts, 
                                                            op_arity - len(cur_args), 
                                                            builder.context_limits, builder.arg_limits)
                    arg_terms = _gen_all_rec(arg_gen_context, cur_counts, cur_arg_counts, at_depth + 1)
                    for arg_i_term, arg_i_counts, arg_i_arg_counts in arg_terms:
                        new_args = [*cur_args, arg_i_term]
                        arg_q.append((new_args, arg_i_counts, arg_i_arg_counts))

        return new_terms

    if start_context is None:
        start_context = builders.default_gen_context

    counts = builders.zero.copy()
    if arg_counts is None:
        arg_counts = counts.copy()

    new_terms_w_counts = _gen_all_rec(start_context, counts, arg_counts, 0)
    new_terms = []
    for new_term, counts, arg_counts in new_terms_w_counts:
        assert np.all(counts >= start_context.min_counts)
        assert np.all(counts <= start_context.max_counts)
        assert (start_context.arg_limits is None) or np.all(arg_counts <= start_context.arg_limits), f"Args counts violaton: {arg_counts} > {start_context.arg_limits}"
        new_terms.append(new_term)

    return new_terms

def grow(builders: Builders,
         grow_depth = 5, grow_leaf_prob: Optional[float] = 0.1,
         rnd: np.random.RandomState = np.random,
         start_context: TermGenContext | None = None,
         arg_counts: np.ndarray | None = None,
         gen_metrics: dict | None = None,
         freq_skew: bool = False
         ) -> Optional[Term]:
    ''' Grow a tree with a given depth '''

    # arity_args = get_arity_args(builders, constraints, default_counts = default_counts)
    term = gen_term(builders, max_depth = grow_depth, 
                    leaf_proba = grow_leaf_prob, rnd = rnd,
                    start_context = start_context,
                    arg_counts = arg_counts, gen_metrics=gen_metrics,
                    freq_skew = freq_skew)
    return term
