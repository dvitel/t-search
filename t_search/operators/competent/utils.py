''' Common utilities for competent operators 
    Operation inversion for Competent Operators - Mutation and Crossover  
    Note that now inv operators work on combinatorial semantics - vector of sets of possible values     

    [s1, s2... sn]. If s1 is empty - any value works, if s1 is None - no value works.    
'''

from bisect import bisect_left
from dataclasses import dataclass, field
from math import prod
import math
from typing import Callable
import torch

from t_search.operators.reduction import LincombMixin
from t_search.syntax import Term, Op
from t_search.syntax.term import TermPos
from t_search.syntax.traverse import postorder_traversal, TRAVERSAL_EXIT_NODE
from t_search.utils import optimize_kb, unique_vector_ids


DesiredSemantics = list[set[float] | None]  # list of sets of possible values for each dimension.

def general_inv(t: DesiredSemantics, args: list[torch.Tensor], arg_i: int, op_inv) -> DesiredSemantics:
    res = []
    for test_id, possible_values in enumerate(t):
        if possible_values is None: 
            res.append(None)
            continue
        elif len(possible_values) == 0:
            res.append(set())
            continue
        else:
            arg_values = [arg[test_id] for arg in args]
            new_desired = set([outcome if outcome is None or math.isfinite(outcome) else None 
                               for desiredValue in possible_values for outcome in op_inv(desiredValue, arg_values, arg_i)])
            if len(new_desired) > 0 and all(d is None for d in new_desired):
                res.append(None)
            else:
                res.append({d for d in new_desired if d is not None})
    return res

def add_inv(desired: float, args: list[float], arg_i: int) -> list[float]:
    res = [ desired - sum(v for i, v in enumerate(args) if i != arg_i ) ]
    return res 

def sub_inv(desired: float, args: list[float], arg_i: int) -> list[float]:
    if arg_i == 0: # desired w.r.t. minuend
        res = [ desired + args[1] ]
    else: # desired w.r.t. subtrahend
        res = [ args[0] - desired ]
    return res

def mul_inv(desired: float, args: list[float], arg_i: int, zero_atol: float = 1e-7) -> list[float]:
    other_mul = prod(v for i, v in enumerate(args) if i != arg_i )
    if abs(other_mul) < zero_atol: 
        if abs(desired) < zero_atol:
            return []
        else:
            return [None]
    else:
        res = [ desired / other_mul ]
        return res
    
def div_inv(desired: float, args: list[float], arg_i: int, zero_atol: float = 1e-7) -> list[float]:
    if arg_i == 0: # desired w.r.t. dividend
        res = [ desired * args[1] ]
    else: # desired w.r.t. divisor
        if abs(desired) < zero_atol:
            return [None]
        else:
            res = [ args[0] / desired ]
    return res
    
def pow_inv(desired: float, args: list[float], arg_i: int, zero_atol: float = 1e-7) -> list[float]:
    base, exponent = args
    if arg_i == 0: # desired w.r.t. base 
        if abs(exponent) < zero_atol:
            if abs(desired - 1) < zero_atol:
                return []
            else:
                return [None]
        elif base < 0 and abs(exponent) < 1:
            return [None]
        else:
            res = [ desired ** (1 / exponent) ]
            return res
    else: # desired w.r.t. exponent
        if (abs(base) < zero_atol and abs(desired) < zero_atol) or (abs(base - 1) < zero_atol and abs(desired - 1) < zero_atol):
            return []
        if base > 0 and desired > 0:
            res = [ math.log(desired) / math.log(base) ]
            return res
        # elif (desired < 0) and round(base) == base:
        #     res = [ -math.log(-desired) / math.log(abs(base)) ]
        return [None]
    
def neg_inv(desired: float, args: list[float], arg_i: int) -> list[float]:
    res = [ -desired ]
    return res

def inv_inv(desired: float, args: list[float], arg_i: int, zero_atol: float = 1e-7) -> list[float]:
    if abs(desired) < zero_atol:
        return [None]
    else:
        res = [ 1 / desired ]
        return res
    
def exp_inv(desired: float, args: list[float], arg_i: int, zero_atol: float = 1e-7) -> list[float]:
    if desired <= 0:
        return [None]
    else:
        res = [ math.log(desired) ]
        return res
    
# NOTE: for log_mod impl should return [exp, -exp] ald op is actually log(abs())
def log_inv(desired: float, args: list[float], arg_i: int) -> list[float]:
    try:
        res = [ math.exp(desired) ]
        return res
    except OverflowError:
        return [None]

# NOTE: only 2 points
def sin_inv(desired: float, args: list[float], arg_i: int) -> list[float]:
    if desired < -1 or desired > 1:
        return [None]
    else:
        asin = math.asin(desired)
        res = [ asin - math.pi - math.pi, asin ]
        return res
    
# NOTE: only 2 points    
def cos_inv(desired: float, args: list[float], arg_i: int) -> list[float]:
    if desired < -1 or desired > 1:
        return [None]
    else:
        acos = math.acos(desired)
        res = [ acos - math.pi - math.pi, acos ]
        return res
    
alg_inv = {
    "add": add_inv,
    "mul": mul_inv,
    "sub": sub_inv,
    "pow": pow_inv,
    "neg": neg_inv,
    "inv": inv_inv,
    "div": div_inv,
    "exp": exp_inv,
    "log": log_inv,
    "sin": sin_inv,
    "cos": cos_inv,    
}

def get_desired_semantics(target: torch.Tensor) -> DesiredSemantics:
    res = []
    for i in range(target.shape[0]):
        val = target[i].item()
        res.append( {val} )
    return res

def invertTerm(term: Term, desired: DesiredSemantics,
                output_getter: Callable[[Term], torch.Tensor],
                op_invs: dict[str, Callable],
                inversion_cache: dict[Term, dict[tuple[Term, int], DesiredSemantics]]) -> None:
    if term.arity() == 0 or not isinstance(term, Op) or term.op_id not in op_invs: # semantic backgpropagation SB works through arguments 
        return
    op_inv = op_invs[term.op_id]
    arg_semantics = output_getter(list(term.get_args()))
    for arg_i, arg in enumerate(term.get_args()):
        arg_desired = general_inv(desired, arg_semantics, arg_i, op_inv)

def pre_invert(root: Term, output_getter: Callable[[Term], torch.Tensor]):
    ''' Tree traversal to collect all occurances and semantics '''

    occurs = {}
    term_arg_semantics: dict[Term, list[torch.Tensor]] = {}
    term_occurs: dict[tuple[Term, int], list[tuple[Term, int]]] = {}

    args_stack = [[]]

    def _enter_args(term: Term, *_):

        cur_occur = occurs.setdefault(term, 0)

        if term not in term_arg_semantics and term.arity() > 0:
            term_arg_semantics[term] = output_getter(list(term.get_args()))

        args_stack[-1].append((term, cur_occur))
        if term.arity() > 0:
            args_stack.append([])
        else:
            occurs[term] += 1
            return TRAVERSAL_EXIT_NODE        
        
    def _exit_term(term: Term, *_):
        cur_occur = occurs[term]
        args = args_stack.pop()
        term_occurs[(term, cur_occur)] = args
        occurs[term] += 1

    postorder_traversal(root, _enter_args, _exit_term)

    return term_arg_semantics, term_occurs 


def backward_desired(root: Term, target: DesiredSemantics, bad_semantics: list[DesiredSemantics],
            output_getter: Callable[[Term], torch.Tensor], get_desired_semantics: Callable[[Term, torch.Tensor], DesiredSemantics],
            op_invs: dict[str, Callable]) -> dict[tuple[Term, int], tuple[DesiredSemantics, list[DesiredSemantics]]]:
    ''' Semantic backpropagation SB'''

    occurs = {}
    term_arg_semantics, term_occurs = pre_invert(root, output_getter)
    
    desired_all = {(root, 0): target}
    undesired_all = {(root, 0): bad_semantics}
    undesired_with_cur_all = {(root, 0): bad_semantics}

    def _enter_args(term: Term, *_):

        if not isinstance(term, Op) or term.op_id not in op_invs:
            # raise ValueError(f"Unsupported term for inversion: {term}")
            return TRAVERSAL_EXIT_NODE

        cur_occur = occurs.setdefault(term, 0)

        desired: DesiredSemantics = desired_all[(term, cur_occur)]
        undesired: list[DesiredSemantics] = undesired_all.get((term, cur_occur), [])
        arg_semantics = term_arg_semantics[term]
        op_inv = op_invs[term.op_id]
        arg_occurs = term_occurs[(term, cur_occur)]

        for arg_i, arg_occur in enumerate(arg_occurs):
            arg_desired = general_inv(desired, arg_semantics, arg_i, op_inv)
            desired_all[arg_occur] = arg_desired        
            arg_undesired_all = []
            for bad_sem in undesired:
                arg_undesired = general_inv(bad_sem, arg_semantics, arg_i, op_inv)
                arg_undesired_all.append(arg_undesired)
            undesired_all[arg_occur] = arg_undesired_all
            arg_cur = get_desired_semantics(arg_occur[0], arg_semantics[arg_i])
            undesired_with_cur_all[arg_occur] = [*arg_undesired_all, arg_cur]
        
    def _exit_term(term: Term, *_):
        occurs[term] += 1

    postorder_traversal(root, _enter_args, _exit_term)

    final = {k: (v, undesired_with_cur_all.get(k, [])) for k,v in desired_all.items()}

    return final 

def get_best_constant(desired: DesiredSemantics, epsilon: float = 1e-5) -> float | None:
    ''' Detect constant semantics that can approximate desired semantics '''
    if any(d is None for d in desired):
        return None
    desired_nonempty = [d for d in desired if len(d) > 0]
    if len(desired_nonempty) == 0:
        return 0.0 # any constant works
    ds = [sorted(d) for d in desired_nonempty ]
    all_vals = set(v for d in ds for v in d)
    for v in all_vals:
        all_close = True
        for d in ds:
            pos = bisect_left(d, v)
            if pos == 0: 
                if abs(d[pos] - v) > epsilon: 
                    all_close = False
                    break
            elif pos == len(d):
                if abs(d[pos - 1] - v) > epsilon:
                    all_close = False
                    break
            else:
                if abs(d[pos] - v) > epsilon and abs(d[pos - 1] - v) > epsilon:
                    all_close = False
                    break
        if all_close:
            return v 
    return None

def get_best_semantics(desired: DesiredSemantics, undesired: list[DesiredSemantics], all_semantics: torch.Tensor, epsilon: float = 1e-5):
    assert len(desired) > 0

    if any(d is None for d in desired): # unsat desired at position 
        return [], None

    # if all(len(d) == 0 for d in desired): # any term will work
    #     return None 

    forbidden_mask = torch.zeros((all_semantics.shape[0],), dtype=torch.bool, device=all_semantics.device)

    for forbidden in undesired:

        if any(d is None for d in forbidden):
            continue # unsat undesired - skip
        
        forbidden_close_mask = torch.ones((all_semantics.shape[0],), dtype=torch.bool, device=all_semantics.device)

        for test_id, forbit_values in enumerate(forbidden):
            if len(forbit_values) == 0:
                continue 
            sem_values = all_semantics[:, test_id].unsqueeze(-1) # (num_terms, 1)
            forbidden_tensor = torch.tensor(list(forbit_values), dtype=all_semantics.dtype, device=all_semantics.device)
            diffs = torch.abs(sem_values - forbidden_tensor.unsqueeze(0)) # (num_terms, num_forbidden)
            close_mask = torch.any(diffs < epsilon, dim=1) # (num_terms,)
            forbidden_close_mask &= close_mask
            del forbidden_tensor
            if not torch.any(forbidden_close_mask):
                break

        forbidden_mask |= forbidden_close_mask

    # test_ids = [i for i, d in enumerate(desired) if len(d) > 0]
    # selected_semantics = all_semantics[:, test_ids]
    # sem_score = torch.zeros((all_semantics.shape[0],), dtype=all_semantics.dtype, device=all_semantics.device)
    closest_desired = torch.zeros((all_semantics.shape[0], len(desired)), dtype=all_semantics.dtype, device=all_semantics.device)
    # closest_test_ids = []
    for test_id, allowed_values in enumerate(desired):
        if len(allowed_values) == 0: # any allowed
            closest_desired[:, test_id] = all_semantics[:, test_id] # precisse same value
            continue 
        sem_values = all_semantics[:, test_id].unsqueeze(-1) # (num_terms, 1)
        allowed_tensor = torch.tensor(list(allowed_values), dtype=all_semantics.dtype, device=all_semantics.device)
        diffs = torch.abs(sem_values - allowed_tensor.unsqueeze(0)) # (num_terms, num_allowed)
        best_per_term_id = torch.argmin(diffs, dim=1) # (num_terms,)
        closest_desired[:, test_id] = allowed_tensor[best_per_term_id] # (num_terms,)
        del allowed_tensor, diffs
        # sem_score += torch.min(diffs, dim=1).values # (num_terms,) 
        # closest_desired_id = torch.unravel_index(, diffs.shape)
        # closest_desired.append(allowed_tensor[closest_desired_id].item())
        # closest_test_ids.append(test_id)

    allowed_ids = torch.where(~forbidden_mask)[0].tolist()
    if len(allowed_ids) == 0:
        return [], None
    allowed_closest_desired = closest_desired[allowed_ids] # (num_allowed    
    del closest_desired, forbidden_mask

    uniq_desired_ids = unique_vector_ids(allowed_closest_desired, atol=epsilon, rtol=0)
    final_closest_desired = allowed_closest_desired[uniq_desired_ids]
    del allowed_closest_desired

    return allowed_ids, final_closest_desired

    # dists_to_closest_desires = (all_semantics - closest_desired).abs().sum(dim=1) # (num_terms,)
    # dists_to_closest_desires[forbidden_mask] = torch.inf

    # best_sem_id = torch.argmin(dists_to_closest_desires).item()

    # if dists_to_closest_desires[best_sem_id] == torch.inf:
    #     return None, None
    
    # best_closest_desired = closest_desired[best_sem_id].clone()
    # del closest_desired, dists_to_closest_desires, forbidden_mask
    
    # return best_sem_id, best_closest_desired
