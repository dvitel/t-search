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

def check_tape(pos_id: int, tape, 
                    penalties: list[tuple[list[int] | int, float]] = [],
                    buf_n:int = 100, rnd: np.random.RandomState = np.random) -> np.ndarray:    
    if pos_id >= tape.shape[0]:
        new_tape = np.zeros((tape.shape[0] + buf_n, tape.shape[1]), dtype=tape.dtype)
        new_tape[:tape.shape[0]] = tape
        new_part = alloc_tape(tape.shape[1], penalties=penalties, buf_n=buf_n, rnd=rnd)
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

    # def dec(self, op_id: int, is_leaf: bool) -> None:
    #     self.max_counts[op_id] -= 1
    #     # self.context_limits[op_id] -= 1
    #     # self.arg_limits[op_id] -= 1
    #     if is_leaf:
    #         if self.min_leaf_counts is not None and self.min_leaf_counts[op_id] > 0:
    #             self.min_leaf_counts[op_id] -= 1
    #     else:
    #         if self.min_nonleaf_counts is not None and self.min_nonleaf_counts[op_id] > 0:
    #             self.min_nonleaf_counts[op_id] -= 1

    # def inc(self, op_id: int, is_leaf: bool) -> None:
    #     self.max_counts[op_id] += 1
    #     # self.context_limits[op_id] += 1
    #     # self.arg_limits[op_id] += 1
    #     if is_leaf:
    #         if self.min_leaf_counts is not None:
    #             self.min_leaf_counts[op_id] += 1
    #     else:
    #         if self.min_nonleaf_counts is not None:
    #             self.min_nonleaf_counts[op_id] += 1

    # def total_min_nonleaf_count(self):
    #     if self.min_nonleaf_counts is None:
    #         return 0
    #     return self.min_nonleaf_counts.sum()
    
    # def total_min_leaf_count(self):
    #     if self.min_leaf_counts is None:
    #         return 0
    #     return self.min_leaf_counts.sum()
    
    # def copy(self) -> 'TermGenContext':
    #     res = TermGenContext(
    #         min_counts=self.min_counts.copy(),
    #         max_counts=self.max_counts.copy(),
    #         context_limits=self.context_limits.copy(),
    #         arg_limits=self.arg_limits
    #     )
    #     return res
    
    # def supports_leaf(self, leaf_ids, nonleaf_ids):
    #     if self.leaf_min_counts is None:
    #         leaf_min_counts = self.min_counts[leaf_ids].sum()
    #         nonleaf_min_counts = self.min_counts[nonleaf_ids].sum()
    #         self._supports_leaf = (leaf_min_counts == 1) and (nonleaf_min_counts == 0)
    #     return self._supports_leaf
    
    # def sat(self, counts: np.ndarray) -> bool:
        
    #     min_sat = np.all(counts >= self.min_counts)
    #     if not min_sat:
    #         return False
    #     max_sat = np.all(self.max_counts >= counts)
    #     if not max_sat:
    #         return False
    #     context_sat = np.all(self.context_limits >= counts)
    #     if not context_sat:
    #         return False
        
    #     return True

    # def sat_args(self, arg_counts: np.ndarray) -> bool:        
        
    #     arg_sat = np.all(self.arg_limits >= arg_counts)
    #     if not arg_sat:
    #         return False
    #     return True

# @dataclass(eq=False, unsafe_hash=False)
@dataclass(frozen=False, eq=False, unsafe_hash=False)
class TermPos:
    term: Term
    occur: int = 0
    pos: int = 0 # pos in parent args
    at_depth: int = 0
    # depth: int = 0
    # size: int = 0
    parent: Optional['TermPos'] = None
    sibling_count: np.ndarray | None = None

global_gen_id = 0 # for debugging

def gen_term(builders: Builders, 
            max_depth = 5, leaf_proba: float | None = 0.1,
            rnd: np.random.RandomState = np.random, buf_n = 100, inf = 100,
            start_context: TermGenContext | None = None,
            arg_counts: np.ndarray | None = None,
            gen_metrics: dict | None = None,
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

    tape = alloc_tape(len(builders), penalties=penalties, buf_n=buf_n, rnd=rnd) # tape is 2d ndarray: (t, score)

    pos_id = 0 
    def get_next_tape_values():
        nonlocal tape, pos_id 
        tape = check_tape(pos_id, tape, penalties=penalties, buf_n=buf_n, rnd=rnd)
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
                op_id_ids, = np.where(gen_context.min_leaf_counts == 1)
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
                    op_id_ids, = np.where(gen_context.min_leaf_counts == 1)
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

def grow(builders: Builders,
         grow_depth = 5, grow_leaf_prob: Optional[float] = 0.1,
         rnd: np.random.RandomState = np.random,
         start_context: TermGenContext | None = None,
         arg_counts: np.ndarray | None = None,
         gen_metrics: dict | None = None,
         ) -> Optional[Term]:
    ''' Grow a tree with a given depth '''

    # arity_args = get_arity_args(builders, constraints, default_counts = default_counts)
    term = gen_term(builders, max_depth = grow_depth, 
                    leaf_proba = grow_leaf_prob, rnd = rnd,
                    start_context = start_context,
                    arg_counts = arg_counts, gen_metrics=gen_metrics)
    return term

def ramped_half_and_half(builders: Builders,
                        rhh_min_depth = 1, rhh_max_depth = 5, rhh_grow_prob = 0.5,
                        grow_leaf_prob: Optional[float] = 0.1, 
                        rnd: np.random.RandomState = np.random,
                        start_context: TermGenContext | None = None,
                        arg_counts: np.ndarray | None = None,
                        gen_metrics: dict | None = None,
                        ) -> Optional[Term]:
    ''' Generate a population of half full and half grow trees '''
    depth = rnd.randint(rhh_min_depth, rhh_max_depth+1)
    leaf_prob = grow_leaf_prob if rnd.rand() < rhh_grow_prob else 0
    term = grow(builders, grow_depth = depth, grow_leaf_prob = leaf_prob, rnd = rnd,
                    start_context = start_context, arg_counts = arg_counts,
                    gen_metrics = gen_metrics)
    return term

# IDEA: dropout in GP, frozen tree positions which cannot be mutated or crossovered - for later

def get_positions(root: Term, pos_cache: dict[Term, list[TermPos]]) -> list[TermPos]:
    ''' Returns dictionary where keys are all positions in the term and values are references to parent position 
        NOTE: we do not return thee root of the term as TermPos as it does not have parent
    '''

    if root in pos_cache:
        return pos_cache[root]

    positions: list[TermPos] = []
    at_depth = 0

    # arg_stack = [(TermPos(None), [])]
    parent_positions = [None]

    occurs = {}
    def _enter_args(term: Term, term_i, *_):
        nonlocal at_depth
        cur_occur = occurs.setdefault(term, 0)
        parent_pos = parent_positions[-1]
        term_pos = TermPos(term, cur_occur, term_i, at_depth,
                           parent = parent_pos)
        parent_positions.append(term_pos)
        at_depth += 1

    def _exit_term(*_):
        nonlocal at_depth
        at_depth -= 1
        cur_pos = parent_positions.pop()
        # if len(children_pos) > 0:
        #     cur_pos.depth = max(child.depth for child in children_pos) + 1
        # cur_pos.size += sum(child.size for child in children_pos)
        positions.append(cur_pos)
        occurs[cur_pos.term] += 1

    postorder_traversal(root, _enter_args, _exit_term)

    pos_cache[root] = positions

    return positions # last one is the root

def enum_occurs(new_term: Term, some_occurs: dict, fn = lambda *_:()):

    def _enter_new_child(t, *_):
        cur_occur = some_occurs.setdefault(t, 0)        

    def _exit_new_child(t, _, p):
        res = fn(t, some_occurs[t], p)
        some_occurs[t] += 1
        return res

    postorder_traversal(new_term, _enter_new_child, _exit_new_child)

def replace(root: Term,
            get_replacement_fn: Callable[[dict[tuple[Term, int], Optional[Term]]], Term],
            builders: Builders) -> Optional[Term]:

    occurs = {}

    replacement = {}

    def _replace_enter(term: Term, term_i: int, parent: Term):
        cur_occur = occurs.get(term, 0)
        new_term = get_replacement_fn((term, cur_occur), occurs = occurs)
        if new_term is not None:
            if isinstance(new_term, Term):
                if parent is None:
                    replacement[None] = new_term
                    return TRAVERSAL_EXIT
                else:
                    args = parent.get_args()
                    new_parent_term_args = tuple((*args[:term_i], new_term, *args[term_i + 1:]))   
                    builder = builders.get_term_builder(parent)
                    new_parent_term = builder.fn(*new_parent_term_args)
                    replacement[parent] = new_parent_term
                    occurs[term] = occurs.get(term, 0) + 1
                    return TRAVERSAL_EXIT_NODE
            else:
                replacement.clear()
                return TRAVERSAL_EXIT

    def _replace_exit(term: Term, term_i: int, parent: Term):
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
        occurs[term] = occurs.get(term, 0) + 1

    postorder_traversal(root, _replace_enter, _replace_exit)

    return None if len(replacement) == 0 else replacement[None]

def order_positions(positions: list[TermPos],
                        select_node_leaf_prob: Optional[float] = 0.1,
                        rnd: np.random.RandomState = np.random) -> np.ndarray:
    pos_proba = rnd.rand(len(positions))
    if select_node_leaf_prob is not None:
        proba_mod = np.array([select_node_leaf_prob if pos.term.arity() == 0 else (1 - select_node_leaf_prob) for pos in positions ], dtype=float)
        pos_proba *= proba_mod
    pos_proba = 1 - pos_proba
    return np.argsort(pos_proba)

def get_pos_constraints(pos: TermPos, builders: Builders, counts_cache: dict[Term, np.ndarray],
                            pos_context_cache: dict[tuple[Term, int], TermGenContext]) -> TermGenContext:
    ''' Assumes that root was generated under correct constraints'''

    if (pos.term, pos.occur) in pos_context_cache:
        return pos_context_cache[(pos.term, pos.occur)]

    chain_to_root = [pos]
    while chain_to_root[-1].parent is not None:
        chain_to_root.append(chain_to_root[-1].parent)
    
    start_i = None
    for parent_i in range(1, len(chain_to_root)):
        parent = chain_to_root[parent_i]
        if (parent.term, parent.occur) in pos_context_cache:
            start_i = parent_i
            break

    if start_i is None:
        start_i = len(chain_to_root) - 1
        _context = builders.default_gen_context
    else:
        parent = chain_to_root[start_i]
        _context = pos_context_cache[(parent.term, parent.occur)]

    for parent_i in range(start_i, -1, -1):
        parent = chain_to_root[parent_i]
        parent_context = _context

        pos_context_cache[(parent.term, parent.occur)] = parent_context

        parent_builder = builders.get_term_builder(parent.term)

        # parent_counts = get_counts(parent.term, builders, counts_cache)
        # parent_one_hot = np.zeros(len(builders), dtype=np.int8)
        # parent_one_hot[parent_builder.id] = 1

        # assert parent_context.sat_args(parent_one_hot)
        # assert parent_context.sat(parent_counts)

        if parent_i == 0: 
            break
        
        arg = chain_to_root[parent_i - 1]
        arg_min_counts = parent_context.min_counts.copy()
        arg_max_counts = parent_context.max_counts.copy()
        arg_min_counts[parent_builder.id] -= 1
        arg_max_counts[parent_builder.id] -= 1
        if parent_builder.context_limits is not None:
            arg_max_counts = np.minimum(arg_max_counts, parent_builder.context_limits)
        arg_context = TermGenContext(
            min_counts=arg_min_counts,
            max_counts= arg_max_counts,
            arg_limits=parent_builder.arg_limits)
        if parent.term.arity() == 1:
            arg_context.min_counts[arg_context.min_counts < 0] = 0
            arg_context.max_counts[arg_context.max_counts < 0] = 0
            pos_context_cache[(arg.term, arg.occur)] = arg_context
            _context = arg_context
            continue

        child_counts = [ get_counts(child, builders, counts_cache) for child in parent.term.get_args() ]        
        other_counts = sum(cnts for child_i, cnts in enumerate(child_counts) if child_i != arg.pos)

        arg_context.min_counts -= other_counts
        arg_context.min_counts[arg_context.min_counts < 0] = 0
        arg_context.max_counts -= other_counts
        arg_context.max_counts[arg_context.max_counts < 0] = 0

        assert np.all(child_counts[arg.pos] <= arg_context.max_counts)
        assert np.all(child_counts[arg.pos] >= arg_context.min_counts)

        if arg_context.arg_limits is not None:
            arg_builder = builders.get_term_builder(arg.term)
            assert arg_context.arg_limits[arg_builder.id] > 0

        _context = arg_context

    res_context = pos_context_cache[(pos.term, pos.occur)]  

    return res_context

def validate_root(root: Term, *, builders: Builders, counts_cache: dict[Term, np.ndarray],                        
                        root_context: TermGenContext | None = None) -> Optional[Term]:
    
    if root_context is None:
        root_context = builders.default_gen_context
                
    if root_context.arg_limits is not None:
        child_count = builders.zero.copy()
        builder = builders.get_term_builder(root)
        child_count[builder.id] += 1
        
        if np.any(child_count > root_context.arg_limits):
            return None
        
    counts = get_counts(root, builders, counts_cache)
    if np.any(counts > root_context.max_counts) or np.any(counts < root_context.min_counts):
        return None

    return root

def validate_term_tree(root: Term, *, builders: Builders, counts_cache: dict[Term, np.ndarray],                        
                        occurs: dict[Term, int] | None = None,
                        context_cache: dict[tuple[Term, int], TermGenContext] | None = None,
                        start_context: TermGenContext | None = None) -> Optional[Term]:
    
    if start_context is None:
        start_context = builders.default_gen_context

    if occurs is None:
        occurs = {}
    else:
        occurs = occurs.copy()

    pos_context_cache = {}

    is_valid = True 

    current_context_stack = [[start_context]]
    child_stack = [[root]]

    def _validate_enter(term: Term, term_i: int, *_):
        nonlocal is_valid        
        cur_occur = occurs.setdefault(term, 0)
        cur_context = current_context_stack[-1][term_i]
                
        if cur_context.arg_limits is not None:
            children = child_stack[-1]
            child_count = builders.zero.copy()
            for arg in children:
                builder = builders.get_term_builder(arg)
                child_count[builder.id] += 1
            
            if np.any(child_count > cur_context.arg_limits):
                is_valid = False
                return TRAVERSAL_EXIT
        

        counts = get_counts(term, builders, counts_cache)
        if np.any(counts > cur_context.max_counts) or np.any(counts < cur_context.min_counts):
            is_valid = False
            return TRAVERSAL_EXIT
        
        pos_context_cache[(term, cur_occur)] = cur_context

        if term.arity() == 0: # leaf
            occurs[term] += 1
            return TRAVERSAL_EXIT_NODE

        term_builder = builders.get_term_builder(term)

        arg_min_counts = cur_context.min_counts.copy()
        arg_max_counts = cur_context.max_counts.copy()
        arg_min_counts[term_builder.id] -= 1
        arg_max_counts[term_builder.id] -= 1
        if term_builder.context_limits is not None:
            arg_max_counts = np.minimum(arg_max_counts, term_builder.context_limits)

        child_stack.append(term.get_args())

        if term.arity() == 1:
            arg_context = TermGenContext(
                min_counts=arg_min_counts,
                max_counts=arg_max_counts,
                arg_limits=term_builder.arg_limits)
            arg_context.min_counts[arg_context.min_counts < 0] = 0
            arg_context.max_counts[arg_context.max_counts < 0] = 0
            current_context_stack.append([arg_context])
            return 
                    
        child_counts = [ get_counts(child, builders, counts_cache) for child in term.get_args() ]
        new_children = []
        term_arity = term.arity()
        for i in range(term_arity):
            child_context = TermGenContext(
                min_counts=(arg_min_counts if (i == term_arity - 1) else arg_min_counts.copy()),
                max_counts=(arg_max_counts if (i == term_arity - 1) else arg_max_counts.copy()),
                arg_limits=term_builder.arg_limits)

            other_counts = sum(cnts for child_i, cnts in enumerate(child_counts) if child_i != i)

            child_context.min_counts -= other_counts
            child_context.min_counts[child_context.min_counts < 0] = 0
            child_context.max_counts -= other_counts
            child_context.max_counts[child_context.max_counts < 0] = 0
            
            new_children.append(child_context)
            
        current_context_stack.append(new_children)

        pass 

    def _validate_exit(term: Term, *_):        
        current_context_stack.pop() 
        child_stack.pop()
        occurs[term] += 1

    postorder_traversal(root, _validate_enter, _validate_exit)

    if is_valid:
        if context_cache is not None:
            context_cache.update(pos_context_cache)
        return root
    return None

def get_pos_sibling_counts(position: TermPos, builders: Builders) -> np.ndarray:
    if position.sibling_count is None:        
        if position.parent is None or position.parent.term.arity() == 1:
            arg_counts = builders.zero.copy()
        else:
            arg_counts = get_immediate_counts(position.parent.term, builders)
            position_builder = builders.get_term_builder(position.term)
            arg_counts[position_builder.id] -= 1
        position.sibling_count = arg_counts
    return position.sibling_count

# def get_parent_path_counts(position: TermPos, builders: Builders) -> np.ndarray:
#     res = builders.zero.copy()
#     cur_pos = position.parent
#     while cur_pos is not None:
#         cur_builder = builders.get_term_builder(cur_pos.term)
#         res[cur_builder.id] += 1
#         cur_pos = cur_pos.parent
#     return res

def one_point_rand_mutation(term: Term,
                            pos_cache: dict[Term, list[TermPos]],
                            pos_context_cache: dict[Term, dict[tuple[Term, int], TermGenContext]],
                            counts_cache: dict[Term, np.ndarray],
                            builders: Builders,
                            rnd: np.random.RandomState = np.random,
                            select_node_leaf_prob: Optional[float] = 0.1,
                            tree_max_depth = 17, max_grow_depth = 5,
                            num_children = 1,
                            mutation_metrics: dict | None = None) -> list[Term]:
    
    # metrics
    success = 0
    fail = 0
    
    positions = get_positions(term, pos_cache)
    pos_contexts = pos_context_cache.setdefault(term, {})

    if len(positions) > 1:
        positions = positions[:-1]

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
        start_context = get_pos_constraints(position, builders, counts_cache, pos_contexts)
        arg_counts = get_pos_sibling_counts(position, builders)

        def _get_replacement_fn(pos, **_):
            if pos == (position.term, position.occur):
                new_term = grow(grow_depth = min(max_grow_depth, tree_max_depth - position.at_depth), rnd = rnd,
                                                          builders = builders, start_context = start_context, arg_counts = arg_counts,
                                                          gen_metrics = mutation_metrics)
                if new_term is None:
                    return TRAVERSAL_EXIT
                return new_term
        mutated_term = replace(term, _get_replacement_fn, builders)
        if mutated_term is not None:       
            # val_poss = get_positions(mutated_term, {})
            # for val_pos in val_poss:
            #     get_pos_constraints(val_pos, builders, {}, {})
            # pass        
            mutants.append(mutated_term)
            success += 1
        else:
            fail += 1

    repr = 0
    if len(mutants) < num_children:
        repr = num_children - len(mutants)
        mutants += [term] * (num_children - len(mutants))

    if mutation_metrics is not None:
        mutation_metrics["success"] = mutation_metrics.get("success", 0) + success
        mutation_metrics["fail"] = mutation_metrics.get("fail", 0) + fail
        mutation_metrics["repr"] = mutation_metrics.get("repr", 0) + repr
        
    return mutants

def get_immediate_counts(root: Term, builders: Builders) -> np.ndarray:

    counts = builders.zero.copy()
    for arg in root.get_args():
        builder = builders.get_term_builder(arg)
        counts[builder.id] += 1

    return counts


def get_counts(root: Term, builders: Builders, counts_cache: dict[Term, np.ndarray]) -> np.ndarray:

    counts_stack = [[]]

    def _enter_args(t: Term, *_):
        if t in counts_cache:
            counts_stack[-1].append(counts_cache[t])
            return TRAVERSAL_EXIT_NODE
        elif t.arity() == 0: # leaf
            builder = builders.get_term_builder(t)
            counts = builders.one_hot[builder.id]
            counts_stack[-1].append(counts)
            return TRAVERSAL_EXIT_NODE
        else:
            counts_stack.append([])

    def _exit_term(t: Term, *_):
        args = counts_stack.pop()
        counts = sum(args)
        builder = builders.get_term_builder(t)
        counts[builder.id] += 1
        counts_cache[t] = counts
        counts_stack[-1].append(counts)

    postorder_traversal(root, _enter_args, _exit_term)

    return counts_stack[-1][-1]

def try_replace_pos(in_term: Term, at_pos: TermPos, with_term: Term, 
                        pos_gen_context: TermGenContext, builders: Builders, 
                        counts_cache: dict[Term, np.ndarray]) -> Optional[Term]:
    
    valid_with_term = validate_root(with_term, builders = builders, root_context = pos_gen_context, counts_cache = counts_cache)
    if valid_with_term is None:
        return None, {}
    
    new_gen_contexts = {}
    def _repl(pos, *, occurs, **_):
        if pos == (at_pos.term, at_pos.occur):
            cur_occur = occurs.get(valid_with_term, 0)
            new_gen_contexts[(valid_with_term, cur_occur)] = pos_gen_context
            return valid_with_term
    new_child = replace(in_term, _repl, builders)
    
    return new_child, new_gen_contexts

def one_point_rand_crossover(term1: Term, term2: Term, *,
                                pos_cache: dict[Term, list[TermPos]],
                                pos_context_cache: dict[Term, dict[tuple[Term, int], TermGenContext]],
                                counts_cache: dict[Term, np.ndarray],
                                depth_cache: dict[Term, int],
                                crossover_cache: dict[tuple[Term, Term, int, Term], Term],
                                builders: Builders,  
                                rnd: np.random.RandomState = np.random,
                                select_node_leaf_prob: Optional[float] = 0.1,
                                exclude_values: bool = True,
                                tree_max_depth = 17,
                                num_children = 1,
                                crossover_metrics: dict | None = None) -> list[Term]:    

    # metrics
    same_subtree = 0
    success = 0 
    fail = 0
    cache_hit = 0

    positions1 = get_positions(term1, pos_cache)
    positions2 = get_positions(term2, pos_cache)
    term1_pos_contexts = pos_context_cache.setdefault(term1, {})
    term2_pos_contexts = pos_context_cache.setdefault(term2, {})

    positions1 = positions1[:-1] # removing root
    positions2 = positions2[:-1]
    num_pairs = len(positions1) * len(positions2)
    if num_pairs > 0:

        pos_ids1 = order_positions(positions1,
                                    select_node_leaf_prob = select_node_leaf_prob, 
                                    rnd = rnd)
        
        if exclude_values:
            pos_ids1 = [pos_id for pos_id in pos_ids1 if not isinstance(positions1[pos_id].term, Value)]

        pos_ids2 = order_positions(positions2,
                                    select_node_leaf_prob = select_node_leaf_prob, 
                                    rnd = rnd)
        
        if exclude_values:
            pos_ids2 = [pos_id for pos_id in pos_ids2 if not isinstance(positions2[pos_id].term, Value)]

    else:
        pos_ids1 = []
        pos_ids2 = []

    children = []

    num_points = min(len(pos_ids1) * len(pos_ids2), num_children)

    for pos_id1, pos_id2 in product(pos_ids1, pos_ids2):
        pos1: TermPos = positions1[pos_id1]
        pos2: TermPos = positions2[pos_id2]
        if pos1.term == pos2.term:
            same_subtree += 2
            continue

        if (term1, pos1.term, pos1.occur, pos2.term) in crossover_cache:
            children.append(crossover_cache[(term1, pos1.term, pos1.occur, pos2.term)])
            cache_hit += 1
        elif pos1.at_depth + get_depth(pos2.term, depth_cache) <= tree_max_depth:   

            pos1_context = get_pos_constraints(pos1, builders, counts_cache, term1_pos_contexts)

            new_child, new_gen_contexts = try_replace_pos(term1, pos1, pos2.term, pos1_context, builders, counts_cache)


            if new_child is not None:
                # val_poss = get_positions(new_child, {})
                # for val_pos in val_poss:
                #     get_pos_constraints(val_pos, builders, {}, {})
                # pass
                children.append(new_child)
                pos_context_cache[new_child] = new_gen_contexts
                crossover_cache[(term1, pos1.term, pos1.occur, pos2.term)] = new_child
                success += 1
            else:
                fail += 1

        if len(children) >= num_children:
            break

        if (term2, pos2.term, pos2.occur, pos1.term) in crossover_cache:
            children.append(crossover_cache[(term2, pos2.term, pos2.occur, pos1.term)])
            cache_hit += 1
        elif pos2.at_depth + get_depth(pos1.term, depth_cache) <= tree_max_depth:

            pos2_context = get_pos_constraints(pos2, builders, counts_cache, term2_pos_contexts)

            new_child, new_gen_contexts = try_replace_pos(term2, pos2, pos1.term, pos2_context, builders, counts_cache)

            if new_child is not None:
                # val_poss = get_positions(new_child, {})
                # for val_pos in val_poss:
                #     get_pos_constraints(val_pos, builders, {}, {})
                # pass                
                children.append(new_child)
                pos_context_cache[new_child] = new_gen_contexts
                crossover_cache[(term2, pos2.term, pos2.occur, pos1.term)] = new_child
                success += 1
            else:
                fail += 1
        
        if len(children) >= num_children:
            break    

    repr = 0
    if len(children) < num_children:
        repr = num_children - len(children)
        left_children = [term1] * (num_children - len(children))
        for i in range(1, len(left_children), 2):
            left_children[i] = term2
        children += left_children

    if crossover_metrics is not None:
        crossover_metrics["same_subtree"] = crossover_metrics.get("same_subtree", 0) + same_subtree
        crossover_metrics["success"] = crossover_metrics.get("success", 0) + success
        crossover_metrics["fail"] = crossover_metrics.get("fail", 0) + fail
        crossover_metrics["cache_hit"] = crossover_metrics.get("cache_hit", 0) + cache_hit
        crossover_metrics["children"] = crossover_metrics.get("children", 0) + len(children)
        crossover_metrics["repr"] = crossover_metrics.get("repr", 0) + repr
        crossover_metrics["num_points"] = crossover_metrics.get("num_points", 0) + num_points

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