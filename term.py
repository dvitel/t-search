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

# @dataclass(eq=False, unsafe_hash=False)
@dataclass(frozen=True, eq=False, unsafe_hash=False)
class TermPos:
    term: Term
    occur: int = 0 # -> id of term occrance in root
    pos: int = 0 # pos in parent args
    at_depth: int = 0
    # depth: int = 0

    # def __eq__(self, other):
    #     if isinstance(other, TermPos):
    #         return self.term == other.term and self.occur == other.occur
    #     return False

    # def __hash__(self):
    #     return hash((self.term, self.occur))    

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

def get_term_pos(term: Term) -> dict[TermPos, TermPos]: 
    ''' Returns dictionary where keys are all positions in the term and values are references to parent position 
        NOTE: we do not return thee root of the term as TermPos as it does not have parent
    '''

    subterms: dict[TermPos, TermPos] = {}
    at_depth = 0
    last_term_pos: dict[Term, TermPos] = {}
    occurs = {}
    def _enter_args(term: Term, term_i, parent: Term):
        nonlocal at_depth
        cur_occur = occurs.setdefault(term, 0)        
        term_pos = TermPos(term, cur_occur, term_i, at_depth)
        last_term_pos[term] = term_pos
        parent_pos = last_term_pos.get(parent, None)
        if parent_pos is not None:
            subterms[term_pos] = parent_pos
        at_depth += 1

    def _exit_term(term: Term, *_):
        nonlocal at_depth
        del last_term_pos[term]
        at_depth -= 1
        occurs[term] += 1

    postorder_traversal(term, _enter_args, _exit_term)

    return subterms

def pick_term_pos(term: Term,
                   pred: Callable[[TermPos, dict[TermPos, TermPos]], tuple[bool, bool]]) -> dict[TermPos, TermPos]:
    ''' Return TermPos that satisfy given predicate. Allows early termination (find_first patern)
        NOTE: we do not return thee root of the term as TermPos as it does not have parent
    '''
    selected_pos = []
    subterms: dict[TermPos, TermPos] = {}
    at_depth = 0
    last_term_pos: dict[Term, TermPos] = {}
    occurs = {}
    def _enter_args(*_):
        nonlocal at_depth
        at_depth += 1

    def _exit_term(term: Term, term_i, parent: Term):
        nonlocal at_depth
        at_depth -= 1
        cur_occur = occurs.setdefault(term, 0)
        term_pos = TermPos(term, cur_occur, term_i, at_depth)
        last_term_pos[term] = term_pos
        parent_pos = last_term_pos.get(parent, None)
        if parent_pos is not None:
            subterms[term_pos] = parent_pos
        should_pick, should_break = pred(term_pos, subterms)
        if should_pick:
            selected_pos.append(term_pos)
        if should_break:
            return TRAVERSAL_EXIT
        occurs[term] += 1

    postorder_traversal(term, _enter_args, _exit_term)

    return subterms, selected_pos

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
    ''' Search for all occurances of pattern in term. 
        * is wildcard leaf. X, Y, Z are meta-variables for non-linear matrching
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

def replace(builders: 'Builders',
            term_pos: TermPos, with_term: Term, term_parents: dict[TermPos, TermPos]) -> Term:
    cur_pos = term_pos
    new_term = with_term
    while cur_pos in term_parents:

        cur_parent = term_parents[cur_pos]

        cur_args = cur_parent.term.get_args()

        new_parent_term_args = tuple((*cur_args[:cur_pos.pos], new_term, *cur_args[cur_pos.pos + 1:]))

        cur_builder = builders.get_builder_for_term(cur_parent.term)
        # assert isinstance(cur_parent.term, Op), f"Expected Op term"
        new_term = cur_builder.fn(*new_parent_term_args) #op_id = cur_parent.term.op_id)
        if new_term is None:
            return None
        cur_pos = cur_parent
        
    return new_term

def evaluate(root: Term, ops: dict[str, Callable],
                get_binding: Callable[[Term, Term], Any] = lambda ti: None,
                set_binding: Callable[[Term, Term, Any], Any] = lambda ti,v:()) -> Any:
    ''' Fully or partially evaluates term (concrete or abstract) '''
    
    # term_occur = {}
    args_stack = [[]]
    def _enter_args(term: Term, *_):
        # cur_occur = term_occur.get(term, 0)        
        res = get_binding(root, term) #, cur_occur))
        if res is not None:
            args_stack[-1].append(res)
            return TRAVERSAL_EXIT_NODE
        args_stack.append([])
        
    def _exit_term(term: Term, *_):
        # cur_occur = term_occur.get(term, 0)
        # term_occur[term] = cur_occur + 1
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

def gen_skeleton(arities: np.ndarray, 
            min_counts: np.ndarray, max_counts: np.ndarray,
            max_depth = 5, leaf_proba: float | None = 0.1,
            rnd: np.random.RandomState = np.random, buf_n = 20,
         ) -> Optional[TermStructure]:
    ''' Arities should be unique and provided in sorted order.
        Counts should correspond to arities 
    '''

    if len(arities) == 0 or arities[0] > 0: # no leaves 
        return None 

    def alloc_tape() -> np.ndarray:
        weights = rnd.random((buf_n, arities.shape[0]))
        if leaf_proba is not None:
            weights[:,0] = np.where(weights[:,0] < leaf_proba, 1, 0)
        return 1 - weights # now smaller is better
    
    tape = alloc_tape() # tape is 2d ndarray: (t, score)

    def get_tape_values(pos_id: int) -> int:   
        nonlocal tape     
        if pos_id >= tape.shape[0]:
            new_tape = np.zeros((tape.shape[0] + buf_n, *tape.shape[1:]), dtype=tape.dtype)
            new_tape[:tape.shape[0]] = tape
            new_part = alloc_tape()
            new_tape[new_tape.shape[0] - buf_n:] = new_part
            tape = new_tape
        return np.copy(tape[pos_id])
    
    def _iter_rec(pos_id: int, min_noleaf_count: int, min_counts: np.ndarray, 
                  max_counts: np.ndarray, at_depth: iter = 0) -> tuple[TermStructure, int] | Literal[-1, 0, 1]:
        if at_depth == max_depth: # attempt only leaves
            if (min_noleaf_count > 0): # cannot satisfy min constraints for non-leaf
                return 0
            if (min_counts[0] > 1): # cannot satisfy min constraints
                return -1
            if max_counts[0] <= 0: # cannot have another leaf 
                return 1
            max_counts[0] -= 1 # no need of min_counts[0] -= 1 as each child has its own min reqs
            new_tree = Leaf
            # print(str(new_tree))
            return new_tree, pos_id + 1
        else:
            inf = 100
            tape_values = get_tape_values(pos_id)
            op_status = np.zeros_like(arities)
            max_count_mask = max_counts <= 0
            tape_values[max_count_mask] = inf # filter out max count violations
            op_status[max_count_mask] = 1 # overflow
            if min_noleaf_count > 0:
                tape_values[0] = inf # cannot have leaves, op should be selected
                op_status[0] = -1 # underflow
            # ordered_ids = np.argsort(tape_values)
            while True:
                op_id = np.argmin(tape_values)
                cur_val = tape_values[op_id]
                # if np.isinf(cur_val):
                if cur_val >= inf: # no more valid ops
                    break
                op_arity = arities[op_id]
                next_pos_id = pos_id + 1
                if op_arity == 0: # leaf selected 
                    if min_counts[0] > 1:
                        op_status[op_id] = -1 
                        tape_values[op_id] = inf
                        continue
                    max_counts[0] -= 1
                    new_tree = Leaf
                    # print(str(new_tree))
                    return new_tree, next_pos_id
                backtrack = None

                new_max_counts = max_counts.copy()
                new_max_counts[op_id] -= 1
                if min_counts[0] == 0 and min_noleaf_count == 0: 
                    def get_min_counts(arg_i):
                        return 0, min_counts 
                else:
                    # left_counts = min_counts.copy()
                    # if left_counts[op_id] > 0:
                    #     left_counts[op_id] -= 1
                    def get_min_counts(arg_i):                        
                        # nonlocal left_counts
                        left_counts = min_counts - (max_counts - new_max_counts)
                        left_counts[left_counts < 0] = 0
                        if arg_i == op_arity - 1:
                            arg_min_nonleaf_count = np.sum(left_counts) - left_counts[0]
                            return arg_min_nonleaf_count, left_counts
                        # new_max_min_counts = np.ceil(left_counts / op_arity)
                        new_max_min_counts = left_counts // (op_arity - arg_i) # left args
                        alloc_counts = rnd.randint(0, new_max_min_counts + 1)
                        arg_min_nonleaf_count = np.sum(alloc_counts) - alloc_counts[0]
                        return arg_min_nonleaf_count, alloc_counts
                # we need to spread min counts between children 
                # new_min_counts = min_counts.copy()
                # new_min_counts[op_id] -= 1
                # new_min_noleaf_count = min_noleaf_count - 1
                arg_op_ids = []
                # print(f"\tB{op_arity}? {at_depth} {min_counts}:{max_counts}")
                for arg_i in range(op_arity):
                    arg_min_nonleaf_count, arg_min_counts = get_min_counts(arg_i)
                    child_opt = _iter_rec(next_pos_id, arg_min_nonleaf_count, 
                                          arg_min_counts, new_max_counts, at_depth + 1)
                    if isinstance(child_opt, int):
                        # print(f"\t<<< {at_depth} {arg_min_counts}:{new_max_counts}")
                        backtrack = child_opt
                        break
                    else:
                        child_op_id, next_pos_id = child_opt
                        arg_op_ids.append(child_op_id)
                if backtrack is not None:
                    if backtrack == -1: # on underflow we can skip all smaller arities
                        smaller_op_ids, = np.where(arities <= op_arity)
                        tape_values[smaller_op_ids] = inf
                        op_status[smaller_op_ids] = backtrack
                    elif backtrack == 1: # on overflow we can skip all larger arities
                        larger_op_ids, = np.where(arities >= op_arity)
                        tape_values[larger_op_ids] = inf
                        op_status[larger_op_ids] = backtrack
                    elif backtrack == 0:
                        tape_values[op_id] = inf
                        op_status[op_id] = backtrack
                    continue
                max_counts[:] = new_max_counts
                new_tree = NonLeafStructure(arg_op_ids)
                # print(str(new_tree))
                return new_tree, next_pos_id
            # print(f"\tfail {op_status}")
            if np.all(op_status == -1):
                return -1 
            elif np.all(op_status == 1):
                return 1
            return 0

    min_nonleaf_count = np.sum(min_counts) - min_counts[0] 
    skeleton, _ = _iter_rec(0, min_nonleaf_count, min_counts.copy(), max_counts.copy())

    return skeleton

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

def gen_term(builders: 'Builders',            
            max_depth = 5, leaf_proba: float | None = 0.1,
            rnd: np.random.RandomState = np.random, buf_n = 100, inf = 100,
         ) -> Optional[Term]:
    ''' Arities should be unique and provided in sorted order.
        Counts should correspond to arities 
    '''

    arity_builders = builders.get_arity_builder_ids()
    if 0 not in arity_builders:
        return None
    leaf_ids = arity_builders[0]

    penalties = [] if leaf_proba is None else [(leaf_ids, leaf_proba)]

    tape = alloc_tape(rnd, penalties=penalties, buf_n=buf_n, rnd=rnd) # tape is 2d ndarray: (t, score)

    def _iter_rec(pos_id: int,
                  min_noleaf_count: int, min_counts: np.ndarray, max_counts: np.ndarray, 
                  context_count: np.ndarray, context_limits: np.ndarray,
                  at_depth: iter = 0) -> tuple[TermStructure, int] | Literal[-1, 0, 1]:
        if at_depth == max_depth: # attempt only leaves
            if min_noleaf_count > 0: # cannot satisfy min constraints for non-leaf
                return 0
            req_id_ids, = np.where(min_counts[leaf_ids] > 0)
            if len(req_id_ids) == 0: # pick any leaf, no leaf min req
                cur_leaf_ids = leaf_ids            
            elif len(req_id_ids) == 1: # need to sat one min 
                cur_leaf_ids = leaf_ids[req_id_ids]
            else: # cannot sat all mins 
                return -1
            selected_id_ids,  = np.where(max_counts[cur_leaf_ids] > 0)
            if len(selected_id_ids) == 0: # cannot have another leaf due to max counts 
                return 1            
            selected_ids = cur_leaf_ids[selected_id_ids]
            context_selected_id_ids, = np.where(context_count[selected_ids] <= context_limits[selected_ids])
            if len(context_selected_id_ids) == 0: # context forbids operators
                return 0
            selected_ids = selected_ids[context_selected_id_ids]
            tape_values = get_tape_values(pos_id, tape, penalties=penalties, buf_n=buf_n, rnd=rnd)
            leaf_tape_values = tape_values[selected_ids]
            while True:
                op_id_id = np.argmin(leaf_tape_values)
                cur_val = tape_values[op_id_id]
                if cur_val >= inf: # no more valid ops
                    break
                selected_id = selected_ids[op_id_id]
                new_term = builders.builders[selected_id].fn() # leaf term
                if new_term is None: # validation failed
                    leaf_tape_values[op_id_id] = inf
                    continue
                context_count[selected_id] += 1
                max_counts[selected_id] -= 1 # no need of min_counts[0] -= 1 as each child has its own min reqs
                print(str(new_term))
                return new_term, pos_id + 1
            return 0 # all validations failed
        else:
            tape_values = get_tape_values(pos_id, tape, penalties=penalties, buf_n=buf_n, rnd=rnd)
            op_status = np.zeros_like(len(builders.builders))
            max_count_mask = max_counts <= 0
            tape_values[max_count_mask] = inf # filter out max count violations
            op_status[max_count_mask] = 1 # overflow
            if min_noleaf_count > 0:
                tape_values[leaf_ids] = inf # cannot have leaves, op should be selected
                op_status[leaf_ids] = -1 # underflow
            # ordered_ids = np.argsort(tape_values)
            while True:
                op_id = np.argmin(tape_values)
                cur_val = tape_values[op_id]
                if cur_val >= inf: # no more valid ops
                    break
                builder = builders.builders[op_id]
                op_arity = builder.arity()
                next_pos_id = pos_id + 1
                if op_arity == 0: # leaf selected 
                    req_id_ids, = np.where(min_counts[leaf_ids] > 0)
                    if len(req_id_ids) == 0:
                        cur_leaf_ids = leaf_ids
                    elif len(req_id_ids) == 1:
                        cur_leaf_ids = leaf_ids[req_id_ids]
                    else:
                        op_status[op_id] = -1 
                        tape_values[op_id] = inf
                        continue
                    new_term = builder.fn() # leaf term
                    if new_term is None: # validation failed
                        op_status[op_id] = 0
                        tape_values[op_id] = inf
                        continue
                    max_counts[selected_id] -= 1
                    print(str(new_term))
                    return new_term, next_pos_id
                backtrack = None

                new_max_counts = max_counts.copy()
                new_max_counts[op_id] -= 1
                if np.all(min_counts[leaf_ids] == 0) and min_noleaf_count == 0: 
                    def get_min_counts(arg_i):
                        return 0, min_counts 
                else:
                    # left_counts = min_counts.copy()
                    # if left_counts[op_id] > 0:
                    #     left_counts[op_id] -= 1
                    def get_min_counts(arg_i):                        
                        # nonlocal left_counts
                        left_counts = min_counts - (max_counts - new_max_counts)
                        left_counts[left_counts < 0] = 0
                        if arg_i == op_arity - 1:
                            arg_min_nonleaf_count = np.sum(left_counts) - np.sum(left_counts[leaf_ids])
                            return arg_min_nonleaf_count, left_counts
                        # new_max_min_counts = np.ceil(left_counts / op_arity)
                        new_max_min_counts = left_counts // (op_arity - arg_i) # left args
                        alloc_counts = rnd.randint(0, new_max_min_counts + 1)
                        arg_min_nonleaf_count = np.sum(alloc_counts) - np.sum(alloc_counts[leaf_ids])
                        return arg_min_nonleaf_count, alloc_counts
                # we need to spread min counts between children 
                # new_min_counts = min_counts.copy()
                # new_min_counts[op_id] -= 1
                # new_min_noleaf_count = min_noleaf_count - 1
                arg_ops = []
                print(f"\tB{op_arity}? {at_depth} {min_counts}:{max_counts}")
                for arg_i in range(op_arity):
                    arg_min_nonleaf_count, arg_min_counts = get_min_counts(arg_i)
                    child_opt = _iter_rec(next_pos_id, arg_min_nonleaf_count, 
                                          arg_min_counts, new_max_counts, at_depth + 1)
                    if isinstance(child_opt, int):
                        print(f"\t<<< {at_depth} {arg_min_counts}:{new_max_counts}")
                        backtrack = child_opt
                        break
                    else:
                        child_op_id, next_pos_id = child_opt
                        arg_ops.append(child_op_id)
                if backtrack is not None:
                    if backtrack == -1: # on underflow we can skip all smaller arities
                        for arity, builder_ids in arity_builders.items():
                            if arity <= op_arity:
                                tape_values[builder_ids] = inf
                                op_status[builder_ids] = backtrack
                    elif backtrack == 1: # on overflow we can skip all larger arities
                        for arity, builder_ids in arity_builders.items():
                            if arity >= op_arity:
                                tape_values[builder_ids] = inf
                                op_status[builder_ids] = backtrack
                    elif backtrack == 0:
                        tape_values[op_id] = inf
                        op_status[op_id] = backtrack
                    continue
                new_tree = builder.fn(*arg_ops)
                if new_tree is None:
                    tape_values[op_id] = inf
                    op_status[op_id] = 0
                    continue
                max_counts[:] = new_max_counts
                print(str(new_tree))
                return new_tree, next_pos_id
            print(f"\tfail {op_status}")
            if np.all(op_status == -1):
                return -1 
            elif np.all(op_status == 1):
                return 1
            return 0

    min_counts = builders.get_min_counts()
    max_counts = builders.get_max_counts()

    min_nonleaf_count = np.sum(min_counts) - np.sum(min_counts[leaf_ids])
    term, _ = _iter_rec(0, min_nonleaf_count, min_counts.copy(), max_counts.copy())

    return term

# arities = np.array([0, 1, 2])
# min_counts = np.array([1, 1, 0])
# max_counts = np.array([3, 3, 1])
# max_depth = 3
# for _ in range(10):
#     s1 = gen_skeleton(arities, min_counts, max_counts, max_depth,
#                         leaf_proba = 0)
#     print("----------------")
# pass    

# @dataclass(frozen=True)
# class TermStructureArgs:
#     builder_ids: list[int]
#     min_counts: np.ndarray
#     max_counts: np.ndarray

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

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class ArityBuilders:
    builders: list['Builder']

    def __post_init__(self):
        self._min_counts: np.ndarray | None = None 
        self._max_counts: np.ndarray | None = None 

    def get_min_counts(self) -> np.ndarray:
        if self._min_counts is None:
            self._min_counts = np.array([b.min_count for b in self.builders])
        return self._min_counts
    
    def get_max_counts(self) -> np.ndarray:
        if self._max_counts is None:
            self._max_counts = np.array([b.max_count for b in self.builders])
        return self._max_counts

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class BuilderAritites:
    arities: np.ndarray
    min_counts: np.ndarray
    max_counts: np.ndarray

UNBOUND = 1000000    

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class Builder:
    fn: Callable
    term_arity: int
    min_count: int = 0
    max_count: int = UNBOUND 

    def __post_init__(self):
        self.bound_id: int | None = None

    def arity(self) -> int:
        return self.term_arity
        # if self._arity is None:
        #     self._arity = get_fn_arity(self.fn)
        # return self._arity

    def is_unbound(self) -> bool:
        return self.min_count == 0 and self.max_count == UNBOUND

class Builders:

    def __init__(self, builders: list[Builder], get_builder_for_term: Callable[[Term], Builder]):
        self.builders: list[Builder] = builders
        self.get_builder_for_term: Callable[[Term], Builder] = get_builder_for_term
        self._bound = []
        self._unbound = []
        for b in self.builders:
            if b.is_unbound():
                self._unbound.append(b)
            else:
                b.bound_id = len(self._bound)
                self._bound.append(b)
        self._arities: BuilderAritites | None = None
        self._min_counts: np.ndarray | None = None # only bound builders
        self._max_counts: np.ndarray | None = None # only bound builders
        self._arity_builders: dict[int, ArityBuilders] | None = None
        self._arity_builder_ids: dict[int, np.ndarray] | None = None

    def get_arity_builders(self) -> dict[int, ArityBuilders]:    
        if self._arity_builders is None:
            arity_builders = {}
            for b in self.builders:
                arity_builders.setdefault(b.arity(), []).append(b)

            self._arity_builders = {a: ArityBuilders(arity_builders[a]) for a in sorted(arity_builders.keys())}
            
            pass 
        return self._arity_builders
    
    def get_arity_builder_ids(self) -> dict[int, np.ndarray]:    
        if self._arity_builder_ids is None:
            arity_builders_ids = {}
            for bi, b in enumerate(self.builders):
                arity_builders_ids.setdefault(b.arity(), []).append(bi)

            self._arity_builders_ids = {a: np.array(arity_builders_ids[a]) for a in sorted(arity_builders_ids.keys())}
            
            pass 
        return self._arity_builder_ids    
    
    def get_arities(self) -> BuilderAritites:
        ''' Returns total min and max counts for each arity '''
        if self._arities is None:
            arity_builders = self.get_arity_builders()
            arities = np.array([a for a in arity_builders.keys()])
            min_counts = np.array([np.sum(builders.get_min_counts()) for builders in arity_builders.values()])
            max_counts = np.array([np.sum(builders.get_max_counts()) for builders in arity_builders.values()])
            max_counts[max_counts > UNBOUND] = UNBOUND
            self._arities = BuilderAritites(arities, min_counts, max_counts)
        return self._arities

    def get_min_counts(self) -> np.ndarray: # NOTE: we return only bound builder min counts
        if self._min_counts is None:
            self._min_counts = np.array([b.min_count for b in self._bound])
        return self._min_counts
    
    def get_max_counts(self) -> np.ndarray:
        if self._max_counts is None:
            self._max_counts = np.array([b.max_count for b in self._bound])
        return self._max_counts   
    
    def set_range(self, min_counts: np.ndarray, max_counts: np.ndarray) -> 'Builders':
        ''' Sets new ranges found bound builders '''
        if len(self._bound) == 0:
            return self 
        new_builders = [b if b.bound_id is None else Builder(b.fn, b.term_arity, 
                                                             min_counts[b.bound_id], 
                                                             max_counts[b.bound_id]) 
                        for b in self.builders ]
        builders = Builders(new_builders, self.get_builder_for_term)
        return builders

    def get_zero_counts(self) -> np.ndarray:
        return np.zeros(len(self._bound), dtype=int)


def instantiate_skeleton(skeleton: TermStructure, builders: Builders,
                         retry_count: int = 2,
                         rnd: np.random.RandomState = np.random) -> Optional[Any]:
    # first we collect list of nodes for each arity 
    arity_terms = {}
    postorder_map(skeleton, lambda t,*_: arity_terms.setdefault(t.arity(), []).append(t),
                  with_cache=False)
    for _ in range(retry_count):
        arity_b = {}
        for arity, arity_builders in builders.get_arity_builders().items():
            terms = arity_terms.get(arity, [])
            # should we check arity count sum() of mins and maxes?? 
            split = _add_factorize(len(terms),
                                min_counts=arity_builders.get_min_counts(),
                                max_counts=arity_builders.get_max_counts(), rnd = rnd)
            if split is None:
                return None
            arity_b_builders = [b_inst for bi, b in enumerate(arity_builders.builders)
                        for b_inst in [b] * split[bi]]
            rnd.shuffle(arity_b_builders)
            arity_b[arity] = arity_b_builders

        occurs = {}

        def _init_term(term: TermStructure, args):
            arity = term.arity()
            cur_occur = occurs.setdefault(arity, 0)
            arity_b_builders = arity_b[arity]
            attempted_builders = set()
            for i in range(cur_occur, len(arity_b_builders)):
                builder = arity_b_builders[i]
                if builder in attempted_builders:
                    continue
                term = builder.fn(*args)
                if term is not None:
                    break
                attempted_builders.add(builder)
            if term is None:
                return None
            if i != cur_occur:
                arity_b_builders[i], arity_b_builders[cur_occur] = arity_b_builders[cur_occur], arity_b_builders[i]
            occurs[arity] += 1
            return term 
        
        instance = postorder_map(skeleton, _init_term, with_cache=False, none_terminate=True)

        if instance is not None:
            break

    return instance

def grow(builders: Builders,
         grow_depth = 5, grow_leaf_prob: Optional[float] = 0.1,
         map_skeleton: Optional[Callable[[TermStructure], TermStructure]] = None,
         rnd: np.random.RandomState = np.random) -> Optional[Term]:
    ''' Grow a tree with a given depth '''

    # arity_args = get_arity_args(builders, constraints, default_counts = default_counts)
    arities = builders.get_arities()
    skeleton = gen_skeleton(arities.arities, arities.min_counts, arities.max_counts, 
                                max_depth = grow_depth, leaf_proba = grow_leaf_prob, rnd = rnd)
    if skeleton is None:
        return None
    
    if map_skeleton is not None:
        skeleton = map_skeleton(skeleton)

    term = instantiate_skeleton(skeleton, builders, rnd = rnd)

    return term

def ramped_half_and_half(builders: Builders,
                        rhh_min_depth = 1, rhh_max_depth = 5, rhh_grow_prob = 0.5,
                        grow_leaf_prob: Optional[float] = 0.1, 
                        map_skeleton: Optional[Callable[[TermStructure], TermStructure]] = None,                     
                        rnd: np.random.RandomState = np.random) -> Optional[Term]:
    ''' Generate a population of half full and half grow trees '''
    depth = rnd.randint(rhh_min_depth, rhh_max_depth+1)
    leaf_prob = grow_leaf_prob if rnd.rand() < rhh_grow_prob else 0
    term = grow(builders, grow_depth = depth, grow_leaf_prob = leaf_prob, map_skeleton = map_skeleton, rnd = rnd)
    return term

# IDEA: dropout in GP, frozen tree positions which cannot be mutated or crossovered - for later

def get_pos_scores(positions: list[TermPos],
                        select_node_leaf_prob: Optional[float] = 0.1,
                        rnd: np.random.RandomState = np.random) -> Optional[np.ndarray]:
    pos_proba = rnd.rand(len(positions))
    if select_node_leaf_prob is not None:
        proba_mod = np.array([select_node_leaf_prob if pos.term.arity() == 0 else (1 - select_node_leaf_prob) for pos in positions ])
        pos_proba *= proba_mod
    return 1 - pos_proba

def order_positions(positions: list[TermPos],
                        select_node_leaf_prob: Optional[float] = 0.1,
                        rnd: np.random.RandomState = np.random) -> list[int]:
    # selecting poss for given number of mutants 
    pos_proba = get_pos_scores(positions, select_node_leaf_prob = select_node_leaf_prob, rnd = rnd)
    ordered_pos_ids = np.argsort(pos_proba).tolist()
    # if len(ordered_pos_ids) > num_positions:
    #     ordered_pos_ids = ordered_pos_ids[:num_positions]
    # elif len(ordered_pos_ids) < num_positions:
    #     repeat_cnt = math.ceil(num_positions / len(ordered_pos_ids))
    #     ordered_pos_ids = (ordered_pos_ids * repeat_cnt)[:num_positions]
    return ordered_pos_ids

def get_counts(root: Term, builders: Builders,
                count_cache: Optional[dict[Term, np.ndarray]] = None) -> np.ndarray:

    if count_cache is None:
        count_cache = {}

    def _enter_args(term: Term, *_):
        if term in count_cache:
            return TRAVERSAL_EXIT_NODE
    
    def _exit_term(term: Term, *_):
        counts = builders.get_zero_counts()
        builder: Builder = builders.get_builder_for_term(term)
        if not builder.is_unbound():
            counts[builder.bound_id] += 1
        if term.arity() > 0:
            counts += sum(count_cache[a] for a in term.get_args())
        count_cache[term] = counts
        pass

    postorder_traversal(root, _enter_args, _exit_term)     

    return count_cache[root]

def one_point_rand_mutation(term: Term, positions: dict[TermPos, TermPos], 
                            builders: Builders,

                            count_cache: dict[Term, np.ndarray] | None = None,
                            rnd: np.random.RandomState = np.random,
                            select_node_leaf_prob: Optional[float] = 0.1,
                            tree_max_depth = 17, max_grow_depth = 5,
                            num_children = 1) -> list[Term]:
    
    term_counts = get_counts(term, builders, count_cache)  

    if len(positions) == 0:
        pos_list = [TermPos(term)] # allow root mutation 
    else:
        pos_list = list(positions.keys())

    ordered_pos_ids = order_positions(pos_list, select_node_leaf_prob = select_node_leaf_prob, rnd = rnd)

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
        position: TermPos = pos_list[pos_id]
        pos_counts = get_counts(position.term, builders, count_cache)
        term_left_counts = term_counts - pos_counts
        pos_min_counts = builders.get_min_counts() - term_left_counts
        pos_min_counts[pos_min_counts < 0] = 0
        pos_max_counts = builders.get_max_counts() - term_left_counts
        pos_max_counts[pos_max_counts < 0] = 0
        cur_builders = builders.set_range(pos_min_counts, pos_max_counts)
        new_child = grow(cur_builders, 
                            grow_depth = min(max_grow_depth, tree_max_depth - position.at_depth), 
                            rnd = rnd)
        if new_child is not None:
            mutated_term = replace(builders, position, new_child, positions)
            if mutated_term is not None:
                mutants.append(mutated_term)
    
    if len(mutants) < num_children:
        mutants += [term] * (num_children - len(mutants))
        
    return mutants

def replacement_counts_sat(pos: TermPos, with_term: Term, root: Term,
                            builders: Builders,
                            count_cache: dict[Term, np.ndarray] | None = None) -> bool:
    root_counts = get_counts(root, builders, count_cache)
    pos_counts = get_counts(pos.term, builders, count_cache)
    with_counts = get_counts(with_term, builders, count_cache)
    new_counts = root_counts - pos_counts + with_counts    

    res = np.all(new_counts >= builders.get_min_counts()) and np.all(new_counts <= builders.get_max_counts())
    return res

def one_point_rand_crossover(term1: Term, term2: Term,
                                           positions1: dict[TermPos, TermPos], positions2: dict[TermPos, TermPos], 
                                builders: Builders,                                
                                depth_cache: dict[Term, int] | None = None,
                                count_cache: dict[Term, np.ndarray] | None = None,
                                rnd: np.random.RandomState = np.random,
                                select_node_leaf_prob: Optional[float] = 0.1,
                                tree_max_depth = 17,
                                num_children = 1):
    
    # term1_depth = get_depth(term1, depth_cache)
    # term2_depth = get_depth(term2, depth_cache)
    # # assert term1_depth <= tree_max_depth
    # # assert term2_depth <= tree_max_depth
    # term1_counts = get_counts(term1, builders, count_cache)
    # term2_counts = get_counts(term2, builders, count_cache)

    if len(positions1) == 0 or len(positions2) == 0: # no crossover
        res = [term1] * num_children
        for i in range(1, num_children, 2):
            res[i] = term2
        return res 
    pos_list1 = list(positions1.keys())
    pos_proba1 = get_pos_scores(pos_list1,
                                select_node_leaf_prob = select_node_leaf_prob, 
                                rnd = rnd)    
    pos_ids1 = np.argsort(pos_proba1)
            
    pos_list2 = list(positions2.keys())
    pos_proba2 = get_pos_scores(pos_list2,
                                select_node_leaf_prob = select_node_leaf_prob, 
                                rnd = rnd)
    
    pos_ids2 = np.argsort(pos_proba2)

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
        pos1 = pos_list1[pos_id1]
        pos2 = pos_list2[pos_id2]
        if (pos1.at_depth + get_depth(pos2.term, depth_cache) <= tree_max_depth) and \
            replacement_counts_sat(pos1, pos2.term, term1, builders, count_cache):
            new_child = replace(builders, pos1, pos2.term, positions1)
            if new_child is not None:
                children.append(new_child)
                if len(children) >= num_children:
                    break
        if (pos2.at_depth + get_depth(pos1.term, depth_cache) <= tree_max_depth) and \
            replacement_counts_sat(pos2, pos1.term, term2, builders, count_cache):
            new_child = replace(builders, pos2, pos1.term, positions2)
            if new_child is not None:
                children.append(new_child)
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