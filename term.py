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


def get_term_repr(term: Term, term_reprs: dict[Term, Term]) -> Term:

    repr_stack = [[]]
    def _find_reprs(t, *_):
        if t in term_reprs:
            repr_stack[-1].append(term_reprs[t])
            return TRAVERSAL_EXIT_NODE
        

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

def alloc_tape(width: int, penalties: list[tuple[list[int] | int, float]] = [],
                buf_n:int = 100, rnd: np.random.RandomState = np.random) -> np.ndarray:
    weights = rnd.random((buf_n, width))
    for ids, p in penalties:
        weights[:,ids] = np.where(weights[:,ids] < p, 1, 0)
    return 1 - weights # now smaller is better

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
    arg_enablence_mask: np.ndarray | None = None
    ''' For each argumemt specifies allowed builders 0/1 bool mask of size (arity, num_builders) '''
    commutative: bool = False

    def __post_init__(self):
        self.id: int | None = None
        if self.term_arity == 1:
            self.commutative = True 

    def arity(self) -> int:
        return self.term_arity
    
    def get_enablence_mask(self, rnd: np.random.RandomState = np.random) -> list[np.ndarray | None]:
        if self.arg_enablence_mask is None:
            return [None] * self.arity()
        if self.commutative:
            res = self.arg_enablence_mask
        else:
            res = self.arg_enablence_mask.copy()
            rnd.shuffle(res)
        return res

class Builders:

    def __init__(self, builders: list[Builder], copy_term: Callable[[Term], Term]):
        self.builders: list[Builder] = builders
        self.copy_term: Callable[[Term], Builder] = copy_term
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

    def disable_arg_builders(self, disabled: dict[Builder, list[list[Builder]]]) -> 'Builders':
        for b, disabled_per_arg in disabled.items():
            mask = np.ones((b.arity(), len(self.builders)), dtype=bool)
            for arg_i in range(b.arity()):
                if arg_i < len(disabled_per_arg):
                    disabled_builders = disabled_per_arg[arg_i]
                    for bi, db in enumerate(self.builders):
                        if db in disabled_builders:
                            mask[arg_i, bi] = False
            b.arg_enablence_mask = mask
        
    def get_leaf_nonleaf_min_counts(self, min_counts: np.ndarray):
        min_leaf_counts = np.sum(min_counts[self.leaf_ids])
        min_nonleaf_counts = np.sum(min_counts[self.nonleaf_ids])
        return (min_leaf_counts, min_nonleaf_counts)

@dataclass(frozen=True, eq=False, unsafe_hash=False)    
class TermGenContext:
    ''' When we generate term, we preserve point requirements for later poitn regeneration '''
    min_leaf_count: int
    min_noleaf_count: int
    min_counts: np.ndarray
    max_counts: np.ndarray
    enablance_mask: np.ndarray | None = None
    # at_depth: int

    def copy(self) -> 'TermGenContext':
        return TermGenContext(self.min_noleaf_count, self.min_counts, self.max_counts.copy(),
                              self.enablance_mask if self.enablance_mask is not None else None)

# @dataclass(eq=False, unsafe_hash=False)
@dataclass(frozen=True, eq=False, unsafe_hash=False)
class TermPos:
    term: Term
    pos: int = 0 # pos in parent args
    at_depth: int = 0
    depth: int = 0
    size: int = 0

    parent: 'TermPos' | None = None  

@dataclass 
class TermGen:    
    term: Term

def gen_term(builders: Builders, 
            max_depth = 5, leaf_proba: float | None = 0.1,
            rnd: np.random.RandomState = np.random, buf_n = 100, inf = 100,
            start_gen_context: TermGenContext | None = None,
            gen_context_cache: Optional[dict[Term, TermGenContext]] = None
         ) -> Optional[Term]:
    ''' Arities should be unique and provided in sorted order.
        Counts should correspond to arities 
    '''

    if gen_context_cache is not None:
        def record_gen(term: Term, gen_context: TermGenContext):
            gen_context_cache[term] = gen_context.copy()
    else:
        def record_gen(term: Term, gen_context: TermGenContext):
            pass

    penalties = [] if leaf_proba is None else [(builders.leaf_ids, leaf_proba)]

    tape = alloc_tape(len(builders), penalties=penalties, buf_n=buf_n, rnd=rnd) # tape is 2d ndarray: (t, score)

    def _iter_rec(pos_id: int, gen_context: TermGenContext,
                  at_depth: iter
                  ) -> tuple[Term, int] | Literal[-1, 0, 1]:
        if at_depth == max_depth: # attempt only leaves
            if gen_context.min_noleaf_count > 0: # cannot satisfy min constraints for non-leaf
                return 0
            if gen_context.enablance_mask is not None:
                enabled_id_ids, = np.where(gen_context.enablance_mask[builders.leaf_ids])
                if len(enabled_id_ids) == 0: # leafs are disabled
                    return 0
                enabled_ids = builders.leaf_ids[enabled_id_ids]
            else:
                enabled_ids = builders.leaf_ids
            # leaf_counts = counts[leaf_ids]
            if gen_context.min_leaf_count > 1: # cannot sat mins
                return -1
            req_id_ids, = np.where(gen_context.min_counts[enabled_ids] > 0)
            selected_id = enabled_ids[req_id_ids][0]
            if gen_context.max_counts[selected_id] == 0: # cannot have another leaf due to max counts 
                return 1            
            new_term = builders.builders[selected_id].fn() # leaf term
            if new_term is None: # validation failed
                return 0
            record_gen(new_term, gen_context)
            gen_context.max_counts[selected_id] -= 1 # no need of min_counts[0] -= 1 as each child has its own min reqs
            # print(str(new_term))
            return new_term, pos_id + 1
        else:
            tape_values = get_tape_values(pos_id, tape, penalties=penalties, buf_n=buf_n, rnd=rnd)
            op_status = np.zeros(len(builders), dtype=int)
            max_count_mask = gen_context.max_counts <= 0
            tape_values[max_count_mask] = inf # filter out max count violations
            if gen_context.enablance_mask is not None:
                tape_values[~gen_context.enablance_mask] = inf # filter out disabled ops
            op_status[max_count_mask] = 1 # overflow
            if (gen_context.min_noleaf_count > 0) or (gen_context.min_leaf_count > 1):
                tape_values[builders.leaf_ids] = inf # cannot have leaves, op should be selected
                op_status[builders.leaf_ids] = -1 # underflow
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
                    record_gen(new_term, gen_context)
                    gen_context.max_counts[op_id] -= 1
                    # print(str(new_term))
                    return new_term, next_pos_id
                backtrack = None

                new_max_counts = gen_context.max_counts.copy()
                new_max_counts[op_id] -= 1
                if builder.context_limits is not None:
                    new_max_counts_w_context = np.minimum(new_max_counts, builder.context_limits)
                    new_max_counts_before = new_max_counts_w_context.copy()
                else:
                    new_max_counts_w_context = new_max_counts
                    new_max_counts_before = gen_context.max_counts

                if gen_context.min_leaf_count == 0 and gen_context.min_noleaf_count == 0: 
                    def get_min_counts(arg_i):
                        return 0, gen_context.min_counts 
                else:
                    # left_counts = min_counts.copy()
                    # if left_counts[op_id] > 0:
                    #     left_counts[op_id] -= 1
                    def get_min_counts(arg_i):                        
                        # nonlocal left_counts
                        left_counts = gen_context.min_counts - (new_max_counts_before - new_max_counts_w_context)
                        left_counts[left_counts < 0] = 0
                        if arg_i == op_arity - 1:
                            min_group_counts = builders.get_leaf_nonleaf_min_counts(left_counts)
                            return min_group_counts, left_counts
                        # new_max_min_counts = np.ceil(left_counts / op_arity)
                        new_max_min_counts = left_counts // (op_arity - arg_i) # left args
                        alloc_counts = rnd.randint(0, new_max_min_counts + 1)
                        min_group_counts = builders.get_leaf_nonleaf_min_counts(alloc_counts)
                        return min_group_counts, alloc_counts
                    
                enabled_masks = builder.get_enablence_mask(rnd = rnd)
                    
                # we need to spread min counts between children 
                arg_ops = []
                # print(f"\t{builder.name}? {at_depth} {min_counts}:{max_counts}")
                for arg_i in range(op_arity):
                    min_group_counts, arg_min_counts = get_min_counts(arg_i)
                    child_gen_context = TermGenContext(*min_group_counts,
                        min_counts = arg_min_counts,
                        max_counts = new_max_counts_w_context,
                        enablance_mask = enabled_masks[arg_i]
                    )
                    child_opt = _iter_rec(next_pos_id, child_gen_context, at_depth + 1)
                    if isinstance(child_opt, int):
                        # print(f"\t<<< {at_depth} {arg_min_counts}:{new_max_counts_w_context}")
                        backtrack = child_opt
                        break
                    else:
                        child_op_id, next_pos_id = child_opt
                        arg_ops.append(child_op_id)
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
                record_gen(new_term, gen_context)
                if builder.context_limits is not None:
                    gen_context.max_counts[:] = gen_context.max_counts - (new_max_counts_before - new_max_counts_w_context)
                else:
                    gen_context.max_counts[:] = new_max_counts
                # print(str(new_tree))
                return new_term, next_pos_id
            # print(f"\tfail {op_status}")
            if np.all(op_status == -1):
                return -1 
            elif np.all(op_status == 1):
                return 1
            return 0

    if start_gen_context is None:
        start_gen_context = TermGenContext(
            *builders.get_leaf_nonleaf_min_counts(builders.min_counts),
            min_counts = builders.min_counts,
            max_counts = builders.max_counts.copy(),
            enablance_mask = None
        )

    term, _ = _iter_rec(0, start_gen_context, 0)

    return term

def grow(builders: Builders,
         grow_depth = 5, grow_leaf_prob: Optional[float] = 0.1,
         rnd: np.random.RandomState = np.random,
         start_gen_context: TermGenContext | None = None,
         gen_context_cache: Optional[dict[Term, TermGenContext]] = None
         ) -> Optional[Term]:
    ''' Grow a tree with a given depth '''

    # arity_args = get_arity_args(builders, constraints, default_counts = default_counts)
    term = gen_term(builders, max_depth = grow_depth, 
                    leaf_proba = grow_leaf_prob, rnd = rnd,
                    start_gen_context = start_gen_context,
                    gen_context_cache = gen_context_cache)
    return term

def ramped_half_and_half(builders: Builders,
                        rhh_min_depth = 1, rhh_max_depth = 5, rhh_grow_prob = 0.5,
                        grow_leaf_prob: Optional[float] = 0.1, 
                        rnd: np.random.RandomState = np.random,
                        start_gen_context: TermGenContext | None = None,
                        gen_context_cache: Optional[dict[Term, TermGenContext]] = None) -> Optional[Term]:
    ''' Generate a population of half full and half grow trees '''
    depth = rnd.randint(rhh_min_depth, rhh_max_depth+1)
    leaf_prob = grow_leaf_prob if rnd.rand() < rhh_grow_prob else 0
    term = grow(builders, grow_depth = depth, grow_leaf_prob = leaf_prob, rnd = rnd,
                    start_gen_context = start_gen_context,
                    gen_context_cache = gen_context_cache)
    return term

# IDEA: dropout in GP, frozen tree positions which cannot be mutated or crossovered - for later

def get_positions(root: Term) -> list[TermPos]:
    ''' Returns dictionary where keys are all positions in the term and values are references to parent position 
        NOTE: we do not return thee root of the term as TermPos as it does not have parent
    '''

    positions: list[TermPos] = []
    at_depth = 0

    parent_poss = [ TermPos(None) ]

    def _enter_args(term: Term, term_i, *_):
        nonlocal at_depth
        parent_pos = parent_poss[-1]
        term_pos = TermPos(term, term_i, at_depth,
                           depth = 0, size = 1, parent = parent_pos)
        parent_poss.append(term_pos)
        at_depth += 1

    def _exit_term(*_):
        nonlocal at_depth
        cur_pos = parent_poss.pop()
        if cur_pos.term.arity() > 0:
            cur_pos.depth += 1
        cur_pos.parent.depth = max(cur_pos.parent.depth, cur_pos.depth)
        cur_pos.parent.size += cur_pos.size
        positions.append(cur_pos)
        at_depth -= 1

    postorder_traversal(root, _enter_args, _exit_term)

    return positions # last one is the root

def replace(builders: Builders,
            term_pos: TermPos, with_term: Term,
            gen_context_cache: dict[Term, TermGenContext]) -> Term:

    cur_pos = term_pos
    new_term = with_term
    cur_gen_context_cache = {}
    while cur_pos.parent.term is not None:

        cur_args = cur_pos.parent.term.get_args()

        new_parent_term_args = tuple((*cur_args[:cur_pos.pos], new_term, *cur_args[cur_pos.pos + 1:]))

        new_term = builders.copy_term(cur_pos.parent.term, new_parent_term_args)
        if new_term is None:
            return None
        cur_gen_context_cache[new_term] = cur_gen_context_cache[cur_pos.parent]
        cur_pos = cur_pos.parent
        
    gen_context_cache.update(cur_gen_context_cache)
    return new_term

def order_positions(positions: list[TermPos],
                        select_node_leaf_prob: Optional[float] = 0.1,
                        rnd: np.random.RandomState = np.random) -> np.ndarray:
    pos_proba = rnd.rand(len(positions))
    if select_node_leaf_prob is not None:
        proba_mod = np.array([select_node_leaf_prob if pos.term.arity() == 0 else (1 - select_node_leaf_prob) for pos in positions ])
        pos_proba *= proba_mod
    pos_proba = 1 - pos_proba
    return np.argsort(pos_proba)

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class TermModificationContext:
    term: Term
    gen_context: TermGenContext
    positions: list[TermPos] | None = None

    def get_term_positions(self) -> list[TermPos]:
        if self.positions is None:
            self.positions = get_positions(self.term)
        return self.positions

def one_point_rand_mutation(term: Term,
                            context: dict[Term, TermModificationContext],
                            builders: Builders,
                            rnd: np.random.RandomState = np.random,
                            select_node_leaf_prob: Optional[float] = 0.1,
                            tree_max_depth = 17, max_grow_depth = 5,
                            num_children = 1) -> list[Term]:
    
    term_context = context[term]
    positions = term_context.get_term_positions()

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
        position_context = context[position.term]
        new_child_gen_context = {}
        new_child = grow(builders, 
                            grow_depth = min(max_grow_depth, tree_max_depth - position.at_depth), 
                            rnd = rnd, start_gen_context = position_context.gen_context,
                            gen_context_cache = new_child_gen_context)
        if new_child is not None:
            mutated_term = replace(builders, position, new_child, new_child_gen_context)
            if mutated_term is not None:
                for t, tg in new_child_gen_context.items():
                    context[t] = TermModificationContext(t, tg)
                mutants.append(mutated_term)
    
    if len(mutants) < num_children:
        mutants += [term] * (num_children - len(mutants))
        
    return mutants

def can_replace(at_pos: TermPos, with_pos: TermPos, 
                at_pos_gen_context: TermGenContext, with_pos_gen_context: TermGenContext,    
                tree_max_depth: int) -> bool:
    
    depth_sat = at_pos.at_depth + with_pos.depth <= tree_max_depth
    if not depth_sat:
        return False
    min_sat = np.all(with_pos_gen_context.min_counts >= at_pos_gen_context.min_counts)
    if not min_sat:
        return False
    max_sat = np.all(with_pos_gen_context.max_counts <= at_pos_gen_context.max_counts)
    if not max_sat:
        return False
    mask_sat = at_pos_gen_context.enablance_mask is None or \
                (with_pos_gen_context.enablance_mask is not None and \
                 np.all(at_pos_gen_context.enablance_mask[with_pos_gen_context.max_counts]))
    return mask_sat

def one_point_rand_crossover(term1: Term, term2: Term,
                                context: dict[Term, TermModificationContext],
                                builders: Builders,  
                                rnd: np.random.RandomState = np.random,
                                select_node_leaf_prob: Optional[float] = 0.1,
                                tree_max_depth = 17,
                                num_children = 1):

    term1_context = context[term1]
    positions1 = term1_context.get_term_positions()
    term2_context = context[term2]
    positions2 = term2_context.get_term_positions()

    if len(positions1) == 1 or len(positions2) == 1: # no crossover of leaves
        res = [term1] * num_children
        for i in range(1, num_children, 2):
            res[i] = term2
        return res 
    pos_ids1 = order_positions(positions1,
                                select_node_leaf_prob = select_node_leaf_prob, 
                                rnd = rnd)    

    pos_ids2 = order_positions(positions2,
                                select_node_leaf_prob = select_node_leaf_prob, 
                                rnd = rnd)

    children = []

    prev_same_count = 0
    prev_len = -1
    max_same_len = min(len(pos_ids1)*len(pos_ids2), 20)
    for pos_id1, pos_id2 in cycle(product(pos_ids1, pos_ids2)):
        if prev_len == len(children):
            prev_same_count += 1
            if prev_same_count > max_same_len:
                break
        else:
            prev_same_count = 0
            prev_len = len(children)
        pos1: TermPos = positions1[pos_id1]
        pos1_gen_context = context[pos1.term].gen_context
        pos2: TermPos = positions2[pos_id2]
        pos2_gen_context = context[pos2.term].gen_context
        if can_replace(pos1, pos2, pos1_gen_context, pos2_gen_context, tree_max_depth):
            new_child_gen_context = {}
            new_child = replace(builders, pos1, pos2.term, new_child_gen_context)
            if new_child is not None:
                children.append(new_child)
                for t, tg in new_child_gen_context.items():
                    context[t] = TermModificationContext(t, tg)
                if len(children) >= num_children:
                    break
        if can_replace(pos2, pos1, pos2_gen_context, pos1_gen_context, tree_max_depth):
            new_child_gen_context = {}
            new_child = replace(builders, pos2, pos1.term, new_child_gen_context)
            if new_child is not None:
                children.append(new_child)
                for t, tg in new_child_gen_context.items():
                    context[t] = TermModificationContext(t, tg)
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