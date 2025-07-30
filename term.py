''' Base utilities to work with term.
    We resort to very simple term representation in a form of tuples (LISP style):
        1. First element is operation, other elements are arguments;
        2. Empty tuple could be see as noop during interpretation, though could be useful during syntactic manipulations.
        3. Operation/leaf is usually a string; op seamntics of symbol is necessary to evaluate the symbol.
        4. Leaves also couls be domain values, e.g. plainly evaluated terms, treated as themselves.
    All additional information, term metrics, should be stored separately.

    Abstract and concrete term forms:
    1. Concrete example: f(x1, x2, x3) = (x1 + x2) * x3 + 4.21 
    2. Abstract form 1: f(x1, x2, x3) = (x1 + x2) * x3 + c, c subject of NLP optimization 
    3. Abstract form 2: f(x1, x2, x3) = (x + x) * x + c, where x, c subject of MINLP optimization
    4. Abstract form 3: f(x1, x2, x3) = (x + x) * x + x, where x is linear combination a1 * x1 + a2 * x2 + a3 * x3 + a4, subject of NLP optimization
'''

from collections import deque
from dataclasses import dataclass, field
from functools import partial
import inspect
from itertools import cycle, product
import math
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import torch

class FnMixin:
    def get_args(self) -> tuple['Term', ...]:
        return self.args
    
    def get_arg(self, i: int):
        return None if i >= len(self.args) else self.args[i]

    def arity(self) -> int:
        return len(self.args)
    
class Term:
    ''' Base class for tree nodes '''

    def get_args(self) -> tuple['Term', ...]:
        return ()
    
    def get_arg(self, i: int):
        return None

    def arity(self) -> int:
        return 0

    
@dataclass(frozen=True, eq=False, unsafe_hash=False, repr=False)
class Op(FnMixin, Term):
    op_id: str

    args: tuple['Term', ...] = field(default_factory=tuple)
    
@dataclass(frozen=True)
class Variable(Term):
    ''' Stores reference to concrete variable '''
    var_id: str
    
@dataclass(frozen=False, eq=False, unsafe_hash=False, repr=False)
class Value(Term):
    ''' Represents constants of target domain 
        Note that constant ref is used, the values are stored separately.
    '''
    value: Any

    def __eq__(self, value):
        if isinstance(value, Value):
            return self.value == value.value
        return False 
    
    def __hash__(self):
        return hash(self.value)
    
@dataclass(frozen=True)
class Wildcard(Term):
    name: str
    
AnyOneWildard = Wildcard(".")    
RepeatWildcard = Wildcard("*")
Wildcards = [AnyOneWildard, RepeatWildcard]

@dataclass(frozen=True, eq=False, unsafe_hash=False, repr=False)
class OpWildcard(Op):
    pass 

def is_ellipsis(term: Term) -> bool:
    return isinstance(term, OpWildcard) and term.op_id == "..."

@dataclass(frozen=True)
class MetaVariable(Term):
    name: str 

@dataclass(frozen=True, eq=False, unsafe_hash=False, repr=False)
class TermStructure(Term):
    ''' Represents tree structure, not a concrete term '''
    pass 

@dataclass(frozen=True, eq=False, unsafe_hash=False, repr=False)
class NonLeafStructure(FnMixin, TermStructure):
    args: tuple[Term, ...] = field(default_factory=tuple)

@dataclass(frozen=True, eq=False, unsafe_hash=False, repr=False)
class LeafStructure(TermStructure):
    pass 

Leaf = LeafStructure()

def parse_float_value(s:str, *_) -> Optional[Term]:
    try:
        return Value(float(s))
    except ValueError:
        return None
    
def parse_meta(s: str, *args) -> Optional[Term]:
    wildcard = next((w for w in Wildcards if w.name == s), None)
    if wildcard is not None:
        return wildcard
    if s.isupper():
        return MetaVariable(s)    
    if s == "...":
        return OpWildcard(s, args)
    return None

def parse_op_or_var(s: str, *args) -> Optional[Term]:
    if len(args) > 0:
        return Op(s, args)
    return Variable(s)

default_parsers = [
    parse_float_value,
    parse_meta,
    parse_op_or_var
]

def name_to_term(name: str, args: Sequence[Term],
                    parsers = default_parsers) -> Term:
    ''' Attempts parsing of a name for creating either var or const. 
        Resorts to func signature at the end.
        op_cache maps arity to name to allocated term_id.
        This is untyped approach where we only consider arity, more complex approach should 
        replace int key of op_cache to TermType dataclass
    '''    
    for parser in parsers:
        term = parser(name, *args)
        if term is not None:
            return term
    return None

TRAVERSAL_EXIT_NODE = 1 
TRAVERSAL_EXIT = 2
    
def postorder_traversal(term: Term, enter_args: Callable, exit_term: Callable):
    ''' enter_args and exit_term are called with entered term and its parent.
        if enter_args returns True, args will be skipped, 
        if exit_term returns True, traversal will be terminated.
    '''
    q = deque([(0, term, 0, None)])
    while len(q) > 0:
        cur_arg_i, cur_term, cur_term_i, cur_parent = q.popleft()
        if cur_arg_i == 0:
            status = enter_args(cur_term, cur_term_i, cur_parent)
            if status == TRAVERSAL_EXIT:
                return
            if status == TRAVERSAL_EXIT_NODE:
                # should_end_traversal = exit_term(cur_term, cur_term_i, cur_parent)
                # if should_end_traversal:
                #     return
                continue
        cur_arg = cur_term.get_arg(cur_arg_i)
        if cur_arg is None:
            status = exit_term(cur_term, cur_term_i, cur_parent)
            if status == TRAVERSAL_EXIT:
                return
        else:            
            q.appendleft((cur_arg_i + 1, cur_term, cur_term_i, cur_parent))
            q.appendleft((0, cur_arg, cur_arg_i, cur_term))


def postorder_map(term: Term, fn: Callable, with_cache = False,
                    none_terminate = False) -> Any:  
    args_stack = [[]]
    term_cache = {}
    if with_cache:
        def add_res(t: Term, res: Any):
            term_cache[t] = res
    else:
        def add_res(*_):
            pass
    def _enter_args(t: Term, *_):
        if t in term_cache:
            processed_t = term_cache[t]
            args_stack[-1].append(processed_t)
            return TRAVERSAL_EXIT_NODE
        args_stack.append([])
    def _exit_term(t: Term, *_):
        term_processed_args = args_stack.pop()
        processed_t = fn(t, term_processed_args)
        add_res(t, processed_t)
        args_stack[-1].append(processed_t) #add to parent args
        if processed_t is None and none_terminate:
            return TRAVERSAL_EXIT
    postorder_traversal(term, _enter_args, _exit_term)
    return args_stack[-1][-1]

def float_formatter(x: Value, *_) -> str:   
    if torch.is_tensor(x.value):
        return f"{x.value.item():.2f}"
    return f"{x.value:.2f}"

default_formatters = {
    Op: lambda t, *args: f"({t.op_id} {' '.join(args)})",
    Variable: lambda t, *_: t.var_id,
    Value: float_formatter,
    NonLeafStructure: lambda t, *args: f"(B{t.arity()} {' '.join(args)})",
    LeafStructure: lambda *_: "L",
    OpWildcard: lambda t, *_: f"({t.op_id} {' '.join([str(a) for a in t.args])})",
    Wildcard: lambda t, *_: t.name,
    MetaVariable: lambda t, *_: t.name,    
}
    
def term_to_str(term: Term, formatters: dict = default_formatters) -> str: 
    ''' LISP style string '''
    def t_to_s(term: Term, args: list[str]):
        if term in formatters:
            return formatters[term](term, *args)
        term_type = type(term)
        if term_type in formatters:
            return formatters[term_type](term, *args)
        name = term_type.__name__        
        return "(" + " ".join([name, *args]) + ")"
    res = postorder_map(term, t_to_s, with_cache=True)
    return res 

Term.__str__ = term_to_str
Term.__repr__ = term_to_str     

# x = str(NonLeafStructure((Leaf, Leaf)))
# pass 

def collect_terms(root: Term, pred: Callable[[Term], bool]) -> list[Term]:
    ''' Find all leaves in root that are equal to term by name '''
    found = []
    def _exit_term(term: Term, *_):
        if pred(term):
            found.append(term)
    postorder_traversal(root, lambda *_: (), _exit_term)
    return found

def get_depth(term: Term, depth_cache: Optional[dict[Term, int]] = None) -> int:
    
    if depth_cache is None:
        depth_cache = {}
    
    def _enter_args(term: Term, *_):
        if (term in depth_cache) or (term.arity() == 0):
            return TRAVERSAL_EXIT_NODE

    def _exit_term(term: Term, *_):
        depth_cache[term] = 1 + max(depth_cache.get(a, 0) for a in term.get_args())

    postorder_traversal(term, _enter_args, _exit_term) 

    return depth_cache.get(term, 0)

def get_size(term: Term, size_cache: Optional[dict[Term, int]] = None) -> int:

    if size_cache is None:
        size_cache = {}

    def _enter_args(term: Term, *_):
        if (term in size_cache) or (term.arity() == 0):
            return TRAVERSAL_EXIT_NODE 

    def _exit_term(term: Term, *_):
        size_cache[term] = 1 + sum(size_cache.get(a, 1) for a in term.get_args())

    postorder_traversal(term, _enter_args, _exit_term)

    return size_cache.get(term, 1)

@dataclass(eq=False, unsafe_hash=False)
class UnifyBindings:
    bindings: dict[str, Term] = field(default_factory=dict)
    renames: dict[str, str] = field(default_factory=dict)

    def copy(self) -> 'UnifyBindings':
        res = UnifyBindings()
        res.bindings = self.bindings.copy()
        res.renames = self.renames.copy()
        return res
    
    def update_with(self, other: 'UnifyBindings'):
        self.bindings.update(other.bindings)
        self.renames.update(other.renames)

    def get(self, *keys) -> tuple[Term, ...]:
        res = tuple(self.bindings.get(self.renames.get(k, k), None) for k in keys)
        return res
    
    def set(self, key: str, value: Term):
        self.bindings[key] = value

    def set_same(self, keys: list[str], to_key: str):
        to_key = self.renames.get(to_key, to_key)
        for k in keys:
            if k != to_key and k not in self.renames:
                self.renames[k] = to_key
    
def _points_are_equiv(ts: Sequence[Term], args: Sequence[Sequence[Term]]) -> bool:
    # arg_counts = [(len(sf), len(s) > 0 and takes_many_args(s[-1]))
    #               for t in ts 
    #               for s in [t.get_args()] 
    #               for sf in [rstrip(s)]]
    # max_count = max(ac for ac, _ in arg_counts)
    first_term = ts[0]
    first_args = args[0]
    def are_same(term1: Term, term2: Term) -> bool:
        if type(term1) != type(term2):
            return False
        if isinstance(term1, Op):
            if term1.op_id != term2.op_id:
                return False
            return True 
        return term1 == term2  # assuming impl of _eq or ref eq     
    res = all(are_same(t, first_term) and \
              (len(a) == len(first_args))
              for t, a in zip(ts, args))
    return res

def set_prev_match(prev_matches: dict[tuple, UnifyBindings | None], 
                   b: UnifyBindings, terms: tuple[Term, ...], match: bool) -> bool:
    prev_matches[terms] = b.copy() if match else None
    return match 

def unify(b: UnifyBindings, *terms: Term,
            prev_matches: dict[tuple, UnifyBindings | None]) -> bool:
    ''' Unification of terms. Uppercase leaves are meta-variables, 

        Note: we do not check here that bound meta-variables recursivelly resolve to concrete terms.
        This should be done by the caller.
    '''
    # if len(terms) == 2 and \
    #     (terms[0].arity() > 0) and (terms[1].arity() > 0) and \
    #     terms[0].op_id == terms[1]/op: # UnderWildcard check 
    #     args1 = terms[0].get_args()
    #     args2 = terms[1].get_args()
    #     if len(args1) > 1 and args1[0] == UnderWildcard:
    #         new_term = terms[1]
    #         new_pat = args[]

    if terms in prev_matches:
        m = prev_matches[terms]
        if m is not None:
            b.update_with(m)
            return True
        return False
    filtered_terms = [t for t in terms if t != AnyOneWildard]    
    if len(filtered_terms) < 2:
        return set_prev_match(prev_matches, b, terms, True)
    if any(t == RepeatWildcard for t in terms):
        return set_prev_match(prev_matches, b, terms, False)
    if len(filtered_terms) == 2:
        el_i = next((i for i, t in enumerate(filtered_terms) if is_ellipsis(t)), -1)
        if el_i >= 0:
            el_term = filtered_terms[el_i]
            if len(el_term.args) == 0:
                return set_prev_match(prev_matches, b, terms, False)
            other_term = filtered_terms[1 - el_i]
            new_pattern = el_term.args[-1]
            if len(el_term.args) > 1:
                name_var = el_term.args[0]
                if not(isinstance(name_var, Variable) and \
                    isinstance(other_term, Op) and \
                    (other_term.op_id == name_var.var_id)):
                    return set_prev_match(prev_matches, b, terms, False)
                matches = []
                for arg in other_term.get_args():
                    matches = match_terms(arg, new_pattern,
                                        with_bindings=b, first_match=True, 
                                        traversal="top_down",
                                        prev_matches=prev_matches)
                    if len(matches) > 0:
                        break 
                return set_prev_match(prev_matches, b, terms, len(matches) > 0)
            else:
                matches = match_terms(other_term, new_pattern, 
                                    with_bindings=b, first_match=True, 
                                    traversal="top_down",
                                    prev_matches=prev_matches)
                return set_prev_match(prev_matches, b, terms, len(matches) > 0)
    t_is_meta = [isinstance(t, MetaVariable) for t in filtered_terms]
    meta_operators = set([t.name for t, is_meta in zip(filtered_terms, t_is_meta) if is_meta])
    meta_terms = b.get(*meta_operators)
    bound_meta_terms = [bx for bx in meta_terms if bx is not None]
    concrete_terms = [t for t, is_meta in zip(filtered_terms, t_is_meta) if not is_meta]
    all_concrete_terms = bound_meta_terms + concrete_terms

    # expanding * wildcards
    all_concrete_terms_args = [t.get_args() for t in all_concrete_terms]
    max_len = max(len(args) for args in all_concrete_terms_args)
    first_repeats = [next((i for i, a in enumerate(args) if a == RepeatWildcard), -1)
                      for args in all_concrete_terms_args]
    expanded_args = [args if ri <= 0 else (args[:ri-1] + (args[ri-1],) * (max_len - len(args) + 2) + args[ri+1:])
                     for args, ri in zip(all_concrete_terms_args, first_repeats)]
    
    expanded_args = [[a for a in args if a != RepeatWildcard] for args in expanded_args]
    
    if len(all_concrete_terms) > 1:
        if not _points_are_equiv(all_concrete_terms, expanded_args):
            return set_prev_match(prev_matches, b, terms, False)
    unbound_meta_operators = [op for op, bx in zip(meta_operators, meta_terms) if bx is None]
    bound_meta_operators = [op for op, bx in zip(meta_operators, meta_terms) if bx is not None]
    if len(unbound_meta_operators) > 0:
        if len(bound_meta_operators) > 0:
            to_key = bound_meta_operators[0]
            b.set_same(unbound_meta_operators, to_key)
        else:
            to_key = unbound_meta_operators[0]
            if len(all_concrete_terms) > 0:
                term = all_concrete_terms[0]
                b.set(to_key, term)
            b.set_same(unbound_meta_operators, to_key)
    if len(all_concrete_terms) >= 2:
        for arg_tuple in zip(*expanded_args):
            if not unify(b, *arg_tuple, prev_matches=prev_matches):
                return set_prev_match(prev_matches, b, terms, False)
    return set_prev_match(prev_matches, b, terms, True)

MatchTraversal = Literal["bottom_up", "top_down"]

def match_terms(root: Term, pattern: Term,
                prev_matches: Optional[dict[tuple, UnifyBindings]] = None,
                with_bindings: UnifyBindings | None = None,
                first_match: bool = False,
                traversal: MatchTraversal = "bottom_up") -> list[tuple[Term, UnifyBindings]]:
    ''' Search for all occurances of pattern in term. 
        * is wildcard leaf. X, Y, Z are meta-variables for non-linear matrching
    '''
    if prev_matches is None:
        prev_matches = {}
    eq_terms = []
    def _match_node(t: Term, *_):
        # if exclude_root and t == root:
        #     return
        if with_bindings is not None:
            bindings = with_bindings.copy()
        else:
            bindings = UnifyBindings()
        if unify(bindings, t, pattern, prev_matches = prev_matches):
            eq_terms.append((t, bindings))
            if first_match:
                if with_bindings is not None:
                    with_bindings.update_with(bindings)
                return TRAVERSAL_EXIT
        pass
    if traversal == "top_down":
        postorder_traversal(root, _match_node, lambda *_: ())
    elif traversal == "bottom_up":
        postorder_traversal(root, lambda *_: (), _match_node)
    else:
        raise ValueError(f"Unknown match traversal: {traversal}")
    return eq_terms

def match_root(root: Term, pattern: Term,
                prev_matches: Optional[dict[tuple, UnifyBindings]] = None) -> Optional[UnifyBindings]:
    ''' Matches root only
    '''
    if prev_matches is None:
        prev_matches = {}
    bindings = UnifyBindings()
    if unify(bindings, root, pattern, prev_matches = prev_matches):
        return bindings
    return None

def skip_spaces(term_str: str, i: int) -> int:
    while i < len(term_str) and term_str[i].isspace():
        i += 1
    return i

def skip_till_break(term_str: str, j: int, breaks) -> int:
    while j < len(term_str) and term_str[j] not in breaks:
        j += 1    
    return j

def parse_literal(term_str: str, i: int = 0): 
    i = skip_spaces(term_str, i)
    # assert i < len(term_str), f"Expected binding or literal at position {i} in term string: {term_str}"
    # if term_str[i] == '[':
    #     j = skip_till_break(term_str, i + 1, "]")
    #     assert j < len(term_str) and term_str[j] == ']', f"Expected ']' at position {j} in term string: {term_str}"
    #     binding_str = term_str[i + 1:j].strip().split(" ")
    #     name = binding_str[0]
    #     values = {int(vs[0]):int(vs[1]) for s in binding_str[1:] if s for vs in [s.strip().split(":")]}        
    #     return (name, values), j + 1
    # else: #literal
    j = skip_till_break(term_str, i + 1, " )")
    literal = term_str[i:j]
    assert literal, f"Literal cannot be empty at position {i}:{j} in term string: {term_str}"
    return literal, j

def parse_term(term_str: str, i: int = 0, parsers = default_parsers) -> tuple[Term, int]:
    ''' Read term from string, return term and end of term after i '''
    branches = deque([[]])
    while True:
        i = skip_spaces(term_str, i)
        if i >= len(term_str):
            break
        if term_str[i] == ')': # end of branch - stop reading args 
            cur_term = branches.popleft() # should contain bindings and args 
            name = cur_term[0]
            args = []
            bindings = {}
            for arg_i in range(1, len(cur_term)):
                arg = cur_term[arg_i]
                if isinstance(arg, Term):
                    args.append(arg)
                elif type(arg) is tuple: 
                    bindings[arg[0]] = arg[1]
            new_term = name_to_term(name, args, parsers = parsers)
            # term = cache_term(term_cache, new_term)
            branches[0].append(new_term)
            i += 1            
        elif term_str[i] == '(': # branch
            literal, i = parse_literal(term_str, i + 1)
            branches.appendleft([literal])
        elif term_str[i] == ':': #binding
            j = skip_till_break(term_str, i+1, " )")
            binding_parts = term_str[i+1:j].split(":")
            branches[0].append((int(binding_parts[0]), int(binding_parts[1])))
            i = j
        else: #leaf
            literal, i = parse_literal(term_str, i)
            # terms.appendleft([binding])
            new_term = name_to_term(literal, [], parsers = parsers)
            # leaf = cache_term(term_cache, new_term)
            branches[0].append(new_term)
    return branches[0][0], i


# def get_term_repr(term: Term, term_reprs: dict[Term, Term]) -> Term:

#     repr_stack = [[]]
#     def _find_reprs(t, *_):
#         if t in term_reprs:
#             repr_stack[-1].append(term_reprs[t])
#             return TRAVERSAL_EXIT_NODE
        

def evaluate(root: Term, ops: dict[str, Callable],
                get_binding: Callable[[Term, Term], Any] = lambda ti: None,
                set_binding: Callable[[Term, Term, Any], Any] = lambda ti,v:()) -> Any:
    ''' Fully or partially evaluates term (concrete or abstract) '''
    
    args_stack = [[]]
    def _enter_args(term: Term, *_):
        res = get_binding(root, term)
        if res is not None:
            args_stack[-1].append(res)
            return TRAVERSAL_EXIT_NODE
        args_stack.append([])
        
    def _exit_term(term: Term, *_):
        args = args_stack.pop()
        res = None
        if isinstance(term, Op) and all(arg is not None for arg in args):
            op_fn = ops[term.op_id]
            res = op_fn(*args)
        if res is not None:            
            set_binding(root, term, res)
        # else:
        #     pass
        args_stack[-1].append(res)

    postorder_traversal(root, _enter_args, _exit_term)

    return args_stack[0][0]    

def alloc_tape(width: int, penalties: list[tuple[list[int] | int, float, float]] = [],
                buf_n:int = 100, rnd: np.random.RandomState = np.random) -> np.ndarray:
    weights = rnd.random((buf_n, width))
    for ids, p, level in penalties:
        selection = weights[:,ids]
        weights[:,ids] = np.where(selection >= p, level, 0)
    return weights # smaller is better

def get_tape_values(pos_id: int, tape, 
                    penalties: list[tuple[list[int] | int, float]] = [],
                    buf_n:int = 100, rnd: np.random.RandomState = np.random) -> int:    
    if pos_id >= tape.shape[0]:
        new_tape = np.zeros((tape.shape[0] + buf_n, tape.shape[1]), dtype=tape.dtype)
        new_tape[:tape.shape[0]] = tape
        new_part = alloc_tape(tape.shape[1], penalties=penalties, buf_n=buf_n, rnd=rnd)
        new_tape[new_tape.shape[0] - buf_n:] = new_part
        tape = new_tape
    return np.copy(tape[pos_id])

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

    counts = np.array(res)

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

UNBOUND = 1000000    

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class Builder:
    name: str
    fn: Callable
    term_arity: int
    min_count: int = 0
    max_count: int = UNBOUND 
    context_limits: np.ndarray | None = None
    ''' Specifies maximum number of builder occurances under this builder,
        int 1d array of size (num_builders,), combines with max_counts eventually 
    '''
    disabled_arg_mask: np.ndarray | None = None
    ''' For each argumemt specifies allowed builders 0/1 bool mask of size (arity, num_builders) '''
    commutative: bool = False

    def __post_init__(self):
        self.id: int | None = None
        if self.term_arity == 1:
            self.commutative = True 

    def arity(self) -> int:
        return self.term_arity
    
    def get_disabled_args(self, rnd: np.random.RandomState = np.random) -> list[np.ndarray | None]:
        if self.disabled_arg_mask is None:
            return [None] * self.arity()
        if self.commutative:
            res = self.disabled_arg_mask
        else:
            res = self.disabled_arg_mask.copy()
            rnd.shuffle(res)
        return res

class Builders:

    def __init__(self, builders: list[Builder], get_term_builder: Callable[[Term], Builder],
                    disallow_initial_leaves: bool = True):
        self.builders: list[Builder] = builders
        self.get_term_builder: Callable[[Term], Builder] = get_term_builder
        self.leaf_ids = []
        self.nonleaf_ids = []
        for bi, b in enumerate(self.builders):
            b.id = bi
            if b.arity() == 0:
                self.leaf_ids.append(bi)
            else:
                self.nonleaf_ids.append(bi)
        self.leaf_ids = np.array(self.leaf_ids)
        self.nonleaf_ids = np.array(self.nonleaf_ids)
        self.min_counts: np.ndarray = np.array([b.min_count for b in self.builders])
        self.max_counts: np.ndarray = np.array([b.max_count for b in self.builders])
        self.arity_builder_ids: dict[int, np.ndarray] = {}
        for bi, b in enumerate(self.builders):
            self.arity_builder_ids.setdefault(b.arity(), []).append(bi)

        self.arity_builder_ids = {a: np.array(self.arity_builder_ids[a]) for a in sorted(self.arity_builder_ids.keys())}

        self.has_context_limits = False
        self.initial_mask = None
        if disallow_initial_leaves:
            self.initial_mask = np.zeros((len(self.builders),), dtype=bool)
            self.initial_mask[self.leaf_ids] = True

        self.min_leaf_counts = np.sum(self.min_counts[self.leaf_ids])
        self.min_nonleaf_counts = np.sum(self.min_counts[self.nonleaf_ids])

        self.default_gen_context = TermGenContext(
            min_leaf_count=self.min_leaf_counts,
            min_nonleaf_count=self.min_nonleaf_counts,
            min_counts=self.min_counts,
            max_counts=self.max_counts,
            disabled_mask=self.initial_mask)
        
        self.leaf_one_hot = {}
        for op_id in self.leaf_ids:
            self.leaf_one_hot[op_id] = np.zeros(len(self.builders), dtype=int)
            self.leaf_one_hot[op_id][op_id] = True        

    def __len__(self):
        return len(self.builders)

    def with_context_limits(self, cl: dict[Builder, dict[Builder, int]]) -> 'Builders':
        for context_builder, limits in cl.items():
            context_limits = np.full(len(self.builders), UNBOUND, dtype=int)
            for bi, b in enumerate(self.builders):
                if b in limits:
                    context_limits[bi] = limits[b]
            context_builder.context_limits = context_limits
        self.has_context_limits = True

    def disabled_arg_builders(self, disabled: dict[Builder, list[list[Builder]]]) -> 'Builders':
        for b, disabled_per_arg in disabled.items():
            mask = np.zeros((b.arity(), len(self.builders)), dtype=bool)
            for arg_i in range(b.arity()):
                if arg_i < len(disabled_per_arg):
                    disabled_builders = disabled_per_arg[arg_i]
                    for bi, db in enumerate(self.builders):
                        if db in disabled_builders:
                            mask[arg_i, bi] = True
            b.disabled_arg_mask = mask
        
@dataclass(frozen=False, eq=False, unsafe_hash=False)    
class TermGenContext:
    ''' When we generate term, we preserve point requirements for later poitn regeneration '''
    min_leaf_count: int
    # leaf_counts: int
    # max_leaf_count: int
    min_nonleaf_count: int
    # nonleaf_counts: int
    # max_nonleaf_count: int
    min_counts: np.ndarray
    # counts: np.ndarray
    max_counts: np.ndarray
    disabled_mask: np.ndarray | None = None
    # at_depth: int

    # def copy(self) -> 'TermGenContext':
    #     return TermGenContext(self.min_leaf_count, self.min_noleaf_count, 
    #                           self.min_counts, self.max_counts.copy(),
    #                           self.enablance_mask if self.enablance_mask is not None else None)

# @dataclass(eq=False, unsafe_hash=False)
@dataclass(frozen=False, eq=False, unsafe_hash=False)
class TermPos:
    term: Term
    occur: int = 0
    pos: int = 0 # pos in parent args
    at_depth: int = 0
    depth: int = 0
    size: int = 0


def gen_term(builders: Builders, 
            max_depth = 5, leaf_proba: float | None = 0.1,
            rnd: np.random.RandomState = np.random, buf_n = 100, inf = 100,
            start_context: TermGenContext | None = None,
            occurs: dict[Term, int] | None = None,
            gen_contexts: Optional[dict[tuple[Term, int], TermGenContext]] = None,
            gen_counts: dict[Term, np.ndarray] | None = None
         ) -> Optional[Term]:
    ''' Arities should be unique and provided in sorted order.
        Counts should correspond to arities 
    '''

    penalties = [] if leaf_proba is None else [(builders.leaf_ids, leaf_proba, 1)]

    tape = alloc_tape(len(builders), penalties=penalties, buf_n=buf_n, rnd=rnd) # tape is 2d ndarray: (t, score)

    def get_occur(term: Term, occurs: dict) -> int:
        cur_occur = occurs.get(term, 0)
        occurs[term] = cur_occur + 1
        return cur_occur
    
    if gen_counts is None:
        gen_counts = {}

    def _iter_rec(pos_id: int, gen_context: TermGenContext,
                  at_depth: iter, occurs: dict[Term, int], counts: np.ndarray,
                  gen_contexts: dict[tuple[Term, int], TermGenContext]
                  ) -> tuple[Term, int] | Literal[-1, 0, 1]:
        op_status = np.zeros(len(builders), dtype=int)
        tape_values = get_tape_values(pos_id, tape, penalties=penalties, buf_n=buf_n, rnd=rnd)            
        if gen_context.disabled_mask is not None:
            tape_values[gen_context.disabled_mask] = inf
            # op_status[gen_context.disabled_mask] = 0
        overflow_mask = (gen_context.max_counts <= 0)
        tape_values[overflow_mask] = inf
        op_status[overflow_mask] = 1 # overflow
        if (gen_context.min_nonleaf_count > 0) or (gen_context.min_leaf_count > 1):
            tape_values[builders.leaf_ids] = inf # cannot have leaves, op should be selected
            op_status[builders.leaf_ids] = -1 # underflow
        if gen_context.min_leaf_count == 1:
            impossiblel_leafs = builders.leaf_ids[gen_context.min_counts[builders.leaf_ids] != 1]
            tape_values[impossiblel_leafs] = inf
            op_status[impossiblel_leafs] = -1
        if at_depth >= max_depth:
            tape_values[builders.nonleaf_ids] = inf
            leaf_status_sum = np.sum(op_status[builders.leaf_ids])
            leaf_count = len(builders.leaf_ids)
            op_status[builders.nonleaf_ids] = 1 if leaf_status_sum == leaf_count else -1 if leaf_status_sum == -leaf_count else 0
        # ordered_ids = np.argsort(tape_values)
        while True:
            op_id = np.argmin(tape_values)
            cur_val = tape_values[op_id]
            if cur_val >= inf: # no more valid ops
                break
            builder = builders.builders[op_id]
            op_arity = builder.arity()
            next_pos_id = pos_id + 1
            if op_arity == 0: # leaf selected - no nonleaf mins
                new_term = builder.fn() # leaf term
                if new_term is None: # validation failed
                    op_status[op_id] = 0
                    tape_values[op_id] = inf
                    continue
                # gen_context.max_counts[op_id] -= 1
                counts[op_id] += 1
                # print(str(new_term))
                cur_occur = get_occur(new_term, occurs)
                gen_contexts[(new_term, cur_occur)] = gen_context
                gen_counts[new_term] = builders.leaf_one_hot[op_id]
                return new_term, next_pos_id
            backtrack = None

            new_counts = np.zeros_like(counts)
            new_counts[op_id] += 1

            if builder.context_limits is not None:
                max_counts = np.minimum(gen_context.max_counts, builder.context_limits)
            else:
                max_counts = gen_context.max_counts
            if op_arity == 1:
                def get_counts(*_):
                    return gen_context.min_leaf_count, gen_context.min_nonleaf_count, \
                            gen_context.min_counts, max_counts
            else:
                bound_ids, = np.where(max_counts != UNBOUND)
                arg_max_counts = max_counts.copy()
                left_max_counts = arg_max_counts[bound_ids] % op_arity
                arg_max_counts[bound_ids] //= op_arity
                if gen_context.min_leaf_count == 0 and gen_context.min_nonleaf_count == 0: 
                    def get_counts(arg_i):
                        nonlocal left_max_counts
                        arg_i_left_max_counts = np.where(left_max_counts > 0, 1, 0)
                        arg_i_max_counts = arg_max_counts.copy()
                        arg_i_max_counts[bound_ids] += arg_i_left_max_counts
                        left_max_counts -= arg_i_left_max_counts
                        return 0, 0, gen_context.min_counts, arg_i_max_counts
                else:
                    def get_counts(arg_i):
                        nonlocal left_max_counts

                        arg_i_left_max_counts = np.where(left_max_counts > 0, 1, 0)
                        arg_i_max_counts = arg_max_counts.copy()
                        arg_i_max_counts[bound_ids] += arg_i_left_max_counts
                        left_max_counts -= arg_i_left_max_counts

                        left_counts = gen_context.min_counts - new_counts
                        left_counts[left_counts < 0] = 0
                        if arg_i == op_arity - 1:
                            arg_i_min_counts = left_counts
                            # min_group_counts = builders.get_leaf_nonleaf_min_counts(left_counts)
                            # return min_group_counts, left_counts
                        else:
                            # new_max_min_counts = left_counts // (op_arity - arg_i) # left args
                            # arg_i_min_counts = rnd.randint(0, new_max_min_counts + 1)
                            arg_i_min_counts = left_counts // (op_arity - arg_i)

                        assert np.all(arg_i_min_counts <= arg_i_max_counts)
                        
                        min_leaf_counts = np.sum(arg_i_min_counts[builders.leaf_ids])
                        min_nonleaf_counts = np.sum(arg_i_min_counts[builders.nonleaf_ids])  

                        return min_leaf_counts, min_nonleaf_counts, arg_i_min_counts, arg_i_max_counts
                
            disabled_args_mask = builder.get_disabled_args(rnd = rnd)
                
            # we need to spread min counts between children 
            arg_ops = []
            # print(f"\t{builder.name}? {at_depth} {gen_context.min_counts}:{gen_context.max_counts}")
            new_occurs = occurs.copy()
            new_gen_contexts = {}
            for arg_i in range(op_arity):
                arg_i_min_leaf_count, arg_i_min_nonleaf_count, arg_i_min_counts, arg_i_max_counts = get_counts(arg_i)
                child_gen_context = TermGenContext(arg_i_min_leaf_count, arg_i_min_nonleaf_count,
                    min_counts = arg_i_min_counts, max_counts = arg_i_max_counts,
                    disabled_mask = disabled_args_mask[arg_i]
                )
                child_opt = _iter_rec(next_pos_id, child_gen_context, at_depth + 1, 
                                      new_occurs, new_counts, new_gen_contexts)
                if isinstance(child_opt, int):
                    # print(f"\t<<< {at_depth} {arg_i_min_counts}:{arg_i_max_counts}")
                    backtrack = child_opt
                    break
                else:
                    child_term, next_pos_id = child_opt
                    arg_ops.append(child_term)
            if backtrack is not None:
                if backtrack == -1: # on underflow we can skip all smaller arities
                    for arity, builder_ids in builders.arity_builder_ids.items():
                        if arity <= op_arity:
                            tape_values[builder_ids] = inf
                            op_status[builder_ids] = backtrack
                elif backtrack == 1: # on overflow we can skip all larger arities
                    for arity, builder_ids in builders.arity_builder_ids.items():
                        if arity >= op_arity:
                            tape_values[builder_ids] = inf
                            op_status[builder_ids] = backtrack
                elif backtrack == 0:
                    tape_values[op_id] = inf
                    op_status[op_id] = backtrack
                continue
            new_term = builder.fn(*arg_ops)
            if new_term is None:
                tape_values[op_id] = inf
                op_status[op_id] = 0
                continue
            counts += new_counts
            gen_counts[new_term] = new_counts
            occurs.update(new_occurs)
            gen_contexts.update(new_gen_contexts)
            cur_occur = get_occur(new_term, occurs)
            gen_contexts[(new_term, cur_occur)] = gen_context
            # print(str(new_term))
            return new_term, next_pos_id
        # print(f"\tfail {op_status}")
        not_nan = np.isfinite(op_status)
        op_s = op_status[not_nan]
        if np.all(op_s == -1):
            return -1 
        elif np.all(op_s == 1):
            return 1
        return 0

    if start_context is None:
        start_context = builders.default_gen_context

    cur_gen_contexts = {}
    cur_occurs = {} if occurs is None else (occurs.copy())

    counts = np.zeros(len(builders), dtype=int)

    res_opt = _iter_rec(0, start_context, 0, cur_occurs, counts, cur_gen_contexts)


    if isinstance(res_opt, int):
        print(f"Fail generate for {start_context.min_counts}:{start_context.max_counts} ")
        return None
    
    new_term, _ = res_opt

    if gen_contexts is not None:
        gen_contexts.update(cur_gen_contexts)

    if occurs is not None:
        occurs.update(cur_occurs)

    assert np.all(counts >= start_context.min_counts)
    assert np.all(counts <= start_context.max_counts)

    return new_term

def grow(builders: Builders,
         grow_depth = 5, grow_leaf_prob: Optional[float] = 0.1,
         rnd: np.random.RandomState = np.random,
         start_context: TermGenContext | None = None,
         occurs: dict[Term, int] | None = None,
         gen_contexts: Optional[dict[tuple[Term, int], TermGenContext]] = None,
         gen_counts: dict[Term, np.ndarray] | None = None
         ) -> Optional[Term]:
    ''' Grow a tree with a given depth '''

    # arity_args = get_arity_args(builders, constraints, default_counts = default_counts)
    term = gen_term(builders, max_depth = grow_depth, 
                    leaf_proba = grow_leaf_prob, rnd = rnd,
                    start_context = start_context, occurs = occurs,
                    gen_contexts = gen_contexts, gen_counts = gen_counts)
    return term

def ramped_half_and_half(builders: Builders,
                        rhh_min_depth = 1, rhh_max_depth = 5, rhh_grow_prob = 0.5,
                        grow_leaf_prob: Optional[float] = 0.1, 
                        rnd: np.random.RandomState = np.random,
                        start_context: TermGenContext | None = None,
                        occurs: dict[Term, int] | None = None,
                        gen_contexts: Optional[dict[tuple[Term, int], TermGenContext]] = None,
                        gen_counts: dict[Term, np.ndarray] | None = None
                        ) -> Optional[Term]:
    ''' Generate a population of half full and half grow trees '''
    depth = rnd.randint(rhh_min_depth, rhh_max_depth+1)
    leaf_prob = grow_leaf_prob if rnd.rand() < rhh_grow_prob else 0
    term = grow(builders, grow_depth = depth, grow_leaf_prob = leaf_prob, rnd = rnd,
                    start_context = start_context, occurs = occurs,
                    gen_contexts = gen_contexts, gen_counts = gen_counts)
    return term

# IDEA: dropout in GP, frozen tree positions which cannot be mutated or crossovered - for later

def get_positions(root: Term) -> list[TermPos]:
    ''' Returns dictionary where keys are all positions in the term and values are references to parent position 
        NOTE: we do not return thee root of the term as TermPos as it does not have parent
    '''

    positions: list[TermPos] = []
    at_depth = 0

    arg_stack = [(TermPos(None), [])]

    occurs = {}
    def _enter_args(term: Term, term_i, *_):
        nonlocal at_depth
        cur_occur = occurs.setdefault(term, 0)
        term_pos = TermPos(term, cur_occur, term_i, at_depth,
                           depth = 0, size = 1)
        arg_stack[-1][-1].append(term_pos)
        arg_stack.append((term_pos, [])) # new args for children
        at_depth += 1

    def _exit_term(*_):
        nonlocal at_depth
        cur_pos, children_pos = arg_stack.pop()
        if len(children_pos) > 0:
            cur_pos.depth = max(child.depth for child in children_pos) + 1
        cur_pos.size += sum(child.size for child in children_pos)
        positions.append(cur_pos)
        at_depth -= 1
        occurs[cur_pos.term] += 1

    postorder_traversal(root, _enter_args, _exit_term)

    return positions # last one is the root

# def validate(term: Term, gen_context: TermGenContext):
#     ''' '''
#     gen_context.


def enum_occurs(new_term: Term, some_occurs: dict, fn = lambda t,o:()) -> int:

    def _enter_new_child(t, *_):
        cur_occur = some_occurs.setdefault(t, 0)        

    def _exit_new_child(t, *_):
        fn(t, some_occurs[t])
        some_occurs[t] += 1

    postorder_traversal(new_term, _enter_new_child, _exit_new_child)

def replace(root: Term, at_pos: tuple[Term, int],
            with_fn: Callable[[dict[tuple[Term, int], int]], Term],
            builders: Builders,
            gen_contexts: dict[tuple[Term, int], TermGenContext]):

    new_contexts = {}

    prev_occurs = {}
    new_occurs = None

    replacement = {}

    def _replace_enter(term: Term, term_i: int, parent: Term):
        nonlocal new_occurs
        cur_occur = prev_occurs.get(term, 0)
        if (term, cur_occur) == at_pos:
            new_occurs = prev_occurs.copy()
            new_term = with_fn(builders = builders, 
                               start_context = gen_contexts[(term, cur_occur)], 
                               occurs = new_occurs, gen_contexts = new_contexts)
            if new_term is not None:
                if parent is None:
                    replacement[None] = new_term
                    return TRAVERSAL_EXIT
                else:
                    enum_occurs(term, prev_occurs)
                    args = parent.get_args()
                    new_parent_term_args = tuple((*args[:term_i], new_term, *args[term_i + 1:]))   
                    builder = builders.get_term_builder(parent)
                    new_parent_term = builder.fn(*new_parent_term_args)
                    replacement[parent] = new_parent_term
                    return TRAVERSAL_EXIT_NODE
            else:
                replacement.clear()
                return TRAVERSAL_EXIT


    def _replace_exit(term: Term, term_i: int, parent: Term):
        nonlocal new_occurs
        new_term = replacement.pop(term, None)
        if new_term is not None:
            if parent is None:
                replacement[None] = new_term
            else:
                args = parent.get_args()
                new_parent_term_args = tuple((*args[:term_i], new_term, *args[term_i + 1:]))   
                builder = builders.get_term_builder(parent)
                new_parent = builder.fn(*new_parent_term_args)
                if new_parent is None:
                    replacement.clear()
                    return TRAVERSAL_EXIT
                replacement[parent] = new_parent
        else:
            new_term = term
        old_occur = prev_occurs.get(term, 0)
        if new_occurs is None:
            new_contexts[(term, old_occur)] = gen_contexts[(term, old_occur)]   
            prev_occurs[term] = old_occur + 1
        else: 
            new_occur = new_occurs.get(new_term, 0)
            new_contexts[(new_term, new_occur)] = gen_contexts[(term, old_occur)]
            prev_occurs[term] = old_occur + 1
            new_occurs[new_term] = new_occur + 1

    postorder_traversal(root, _replace_enter, _replace_exit)

    return ((None if len(replacement) == 0 else replacement[None]), new_contexts)

def order_positions(positions: list[TermPos],
                        select_node_leaf_prob: Optional[float] = 0.1,
                        rnd: np.random.RandomState = np.random) -> np.ndarray:
    pos_proba = rnd.rand(len(positions))
    if select_node_leaf_prob is not None:
        proba_mod = np.array([select_node_leaf_prob if pos.term.arity() == 0 else (1 - select_node_leaf_prob) for pos in positions ])
        pos_proba *= proba_mod
    pos_proba = 1 - pos_proba
    return np.argsort(pos_proba)

def one_point_rand_mutation(term: Term,
                            gen_contexts: dict[Term, dict[tuple[Term, int], TermGenContext]],
                            gen_counts: dict[Term, np.ndarray],
                            pos_cache: dict[Term, list[TermPos]],
                            builders: Builders,
                            rnd: np.random.RandomState = np.random,
                            select_node_leaf_prob: Optional[float] = 0.1,
                            tree_max_depth = 17, max_grow_depth = 5,
                            num_children = 1) -> list[Term]:
    
    cur_gen_context = gen_contexts[term]
    if term not in pos_cache:
        pos_cache[term] = get_positions(term)
    positions = pos_cache[term]

    if len(positions) > 1:
        positions.pop() # remove root

    ordered_pos_ids = order_positions(positions, 
                                      select_node_leaf_prob = select_node_leaf_prob, 
                                      rnd = rnd)

    mutants = []
    prev_same_count = 0
    prev_len = -1
    for pos_id in cycle(ordered_pos_ids):
        if len(mutants) >= num_children:
            break
        if prev_len == len(mutants):
            prev_same_count += 1
            if prev_same_count > len(ordered_pos_ids):
                break
        else:
            prev_same_count = 0
            prev_len = len(mutants)
        position: TermPos = positions[pos_id]
        mutated_term, new_gen_contexts = replace(term, (position.term, position.occur), 
                                            lambda **kwargs: grow(grow_depth = min(max_grow_depth, tree_max_depth - position.at_depth), rnd = rnd,
                                                                  gen_counts = gen_counts, **kwargs), builders, cur_gen_context)
        if mutated_term is not None:            
            mutants.append(mutated_term)
            gen_contexts[mutated_term] = new_gen_contexts
    
    if len(mutants) < num_children:
        mutants += [term] * (num_children - len(mutants))
        
    return mutants

def can_replace(builders: Builders, at_pos: TermPos, at_pos_gen_context: TermGenContext,
                with_pos: TermPos, with_counts: np.ndarray,    
                tree_max_depth: int) -> bool:
    
    depth_sat = at_pos.at_depth + with_pos.depth <= tree_max_depth
    if not depth_sat:
        return False
    min_sat = np.all(with_counts >= at_pos_gen_context.min_counts)
    if not min_sat:
        return False
    max_sat = np.all(with_counts <= at_pos_gen_context.max_counts)
    if not max_sat:
        return False
    with_builder = builders.get_term_builder(with_pos.term)
    mask_sat = at_pos_gen_context.disabled_mask is None or \
                 not at_pos_gen_context.disabled_mask[with_builder.id]
    return mask_sat

def copy_gen_contexts(term, *, occurs, gen_contexts, new_gen_context, **_):
    def _copy_fn(t: Term, o: int):
        gen_contexts[(t, o)] = new_gen_context
    enum_occurs(term, occurs, _copy_fn)
    return term

def get_counts(root: Term, builders: Builders, counts_cache: dict[Term, np.ndarray]) -> np.ndarray:

    counts_stack = [[]]

    def _enter_args(t: Term):
        if t in counts_cache:
            counts_stack[-1].append(counts_cache[t])
            return TRAVERSAL_EXIT_NODE
        elif t.arity() == 0: # leaf
            builder = builders.get_term_builder(t)
            counts = builders.leaf_one_hot[builder.id]
            counts_stack[-1].append(counts)
            return TRAVERSAL_EXIT_NODE
        else:
            counts_stack.append([])

    def _exit_term(t: Term):
        args = counts_stack.pop()
        counts = sum(args)
        counts_cache[t] = counts
        counts_stack[-1].append(counts)

    postorder_traversal(root, _enter_args, _exit_term)

    return counts_stack[-1][-1]

def one_point_rand_crossover(term1: Term, term2: Term,
                                gen_contexts: dict[Term, dict[tuple[Term, int], TermGenContext]],
                                gen_counts: dict[Term, np.ndarray],
                                pos_cache: dict[Term, list[TermPos]],
                                builders: Builders,  
                                rnd: np.random.RandomState = np.random,
                                select_node_leaf_prob: Optional[float] = 0.1,
                                tree_max_depth = 17,
                                num_children = 1):

    term1_gen_context = gen_contexts[term1]
    if term1 not in pos_cache:
        pos_cache[term1] = get_positions(term1)
    positions1 = pos_cache[term1]
    term2_gen_context = gen_contexts[term2]
    if term2 not in pos_cache:
        pos_cache[term2] = get_positions(term2)
    positions2 = pos_cache[term2]

    if len(positions1) == 1 or len(positions2) == 1: # no crossover of leaves
        res = [term1] * num_children
        for i in range(1, num_children, 2):
            res[i] = term2
        return res 
    else:
        positions1 = positions1[:-1] # remove root
        positions2 = positions2[:-1] # remove root
    pos_ids1 = order_positions(positions1,
                                select_node_leaf_prob = select_node_leaf_prob, 
                                rnd = rnd)    

    pos_ids2 = order_positions(positions2,
                                select_node_leaf_prob = select_node_leaf_prob, 
                                rnd = rnd)

    children = []

    for pos_id1, pos_id2 in product(pos_ids1, pos_ids2):
        pos1: TermPos = positions1[pos_id1]
        pos2: TermPos = positions2[pos_id2]
        pos1_gen_context = term1_gen_context[(pos1.term, pos1.occur)]
        if can_replace(builders, pos1, pos1_gen_context, pos2, gen_counts[pos2.term], tree_max_depth):

            copy_fn = partial(copy_gen_contexts, pos2.term, new_gen_context = pos1_gen_context)

            new_child, new_gen_contexts = replace(term1, (pos1.term, pos1.occur), 
                                                    copy_fn, builders, term1_gen_context)
            if new_child is not None:
                children.append(new_child)
                gen_contexts[new_child] = new_gen_contexts
                if len(children) >= num_children:
                    break
        pos2_gen_context = term2_gen_context[(pos2.term, pos2.occur)]
        if can_replace(builders, pos2, pos2_gen_context, pos1, gen_counts[pos1.term], tree_max_depth):

            copy_fn = partial(copy_gen_contexts, pos1.term, new_gen_context = pos2_gen_context)
                        
            new_child, new_gen_contexts = replace(term2, (pos2.term, pos2.occur), 
                                            copy_fn, builders, term2_gen_context)
            if new_child is not None:
                children.append(new_child)
                gen_contexts[new_child] = new_gen_contexts
                if len(children) >= num_children:
                    break

    if len(children) < num_children:
        left_children = [term1] * (num_children - len(children))
        for i in range(1, len(left_children), 2):
            left_children[i] = term2
        children += left_children

    return children

def unique_term(root: Term, term_cache: dict[tuple, Term] | None = None) -> Term:
    ''' Remaps term to unique terms '''
    if term_cache is None:
        term_cache = {}

    def _map(term, args):
        if isinstance(term, Op):
            signature = (term.op_id, *args)
            if signature not in term_cache:
                term_cache[signature] = term
            return term_cache[signature]
        return term

    res = postorder_map(root, _map, with_cache = False)

    return res


if __name__ == "__main__":

    # tests
    t1, _ = parse_term("(f (f X (f x (f x)) (f x (f x))))")
    print(str(t1))
    t1_str1 = term_to_str(t1)
    t2, _ = parse_term("(f (f (f x x) Y Y))")
    t3, _ = parse_term("(f Z)")
    # b = UnifyBindings()
    # res = unify(b, points_are_equiv, t1, t2, t3)
    pass


    t1_str = "(f (f (f x x) (f 1.42 (f x)) (f 1.42 (f x))))"
    # t1_str = "(f x x 1.43 1.42)"
    t1, _ = parse_term(t1_str)

    depth = get_depth(t1)
    print(depth)
    pass    

    print(str(t1))
    assert str(t1) == t1_str, f"Expected {t1_str}, got {str(t1)}"
    pass
    # t1, _ = parse_term("(f x y x x x x x)")
    # t1, _ = parse_term("(inv (exp (mul x (cos (sin (exp (add 0.134 (exp (pow x x)))))))))")
    t1, _ = parse_term("(pow (pow x0 1.81) (log 1.02))")
    # p1, _ = parse_term("(f (f X X) Y Y)")
    # p1, _ = parse_term("(... (f 1.42 X))")
    # p1, _ = parse_term("(exp (... (exp (... (exp .)))))")
    p1, _ = parse_term("(... pow (pow . .))")
    # p1, _ = parse_term("(... exp (exp X))")
    # p1, _ = parse_term("(f x X *)")

    p1_str = term_to_str(p1)
    term_cache = {}
    ut1 = unique_term(t1, term_cache)
    up1 = unique_term(p1, term_cache)
    matches = match_terms(ut1, up1, traversal="bottom_up", first_match=True)
    matches = [(str(m[0]), {k:str(v) for k, v in m[1].bindings.items()}) for m in matches]
    pass

    # res, _ = parse_term("  \n(   f   (g    x :0:1)  (h \nx) :0:12)  \n", 0)
    t1, _ = parse_term("  \n(   f   (g    x)  (h \nx))  \n", 0)
    leaves = collect_terms(t1, lambda t: isinstance(t, Variable))
    # bindings = bind_terms(leaves, 1)
    bindings = {parse_term("x")[0]: 1}
    print(str(t1))
    ev1 = evaluate(t1, { "f": lambda x, y: x + y, 
                         "g": lambda x: x * 2, 
                         "h": lambda x: x ** 2 }, lambda _,x: bindings.get(x), 
                   lambda _,*x: bindings.setdefault(*x))

    pass    