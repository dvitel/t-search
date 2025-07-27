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
from itertools import product
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
    
@dataclass(frozen=False)
class Value(Term):
    ''' Represents constants of target domain 
        Note that constant ref is used, the values are stored separately.
    '''
    value: Any
    
@dataclass(frozen=True)
class Wildcard(Term):
    name: Literal["?", "*"] = "?"
    
StarWildcard = Wildcard("*")
QuestionWildcard = Wildcard("?")    

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

def parse_float(s:str) -> Optional[float]:
    try:
        value = float(s)
    except ValueError:
        value = None
    return value        

def name_to_term(name: str, args: Sequence[Term],
                    parsers: dict = {
                        Value: parse_float
                    }) -> Term:
    ''' Attempts parsing of a name for creating either var or const. 
        Resorts to func signature at the end.
        op_cache maps arity to name to allocated term_id.
        This is untyped approach where we only consider arity, more complex approach should 
        replace int key of op_cache to TermType dataclass
    '''    
    if len(args) == 0:
        if name == "?":
            return QuestionWildcard
        if name == "*":
            return StarWildcard
        if name.isupper():
            return MetaVariable(name)        
        for term_type, parser in parsers.items():
            value = parser(name)
            if value is not None:
                return term_type(value)
        return Variable(name)
    return Op(name, tuple(args))

# def cache_term(term_cache: dict[tuple, Term], term: Term,
#                 cache_cb: Callable = lambda t,s:()) -> Term:
#     '''  Check if term is already present and if so, returns cached version for given term.
#          Untyped approach, more complex method should define signature preciselly. 
         
#          Returns cached instance of term and hit/miss flag
#     '''
    
#     # 3 parts, term type as general category, term_id, 
#     #          uniquelly identifies Node among possible nodes,
#     #          arg refs should be previously cached
#     signature = term.get_signature(up_to="all")
#     if signature in term_cache:
#         term = term_cache[signature]
#         cache_cb(term, True)
#     else:
#         term_cache[signature] = term
#         cache_cb(term, False)
#     return term

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


def postorder_map(term: Term, fn: Callable, with_cache = False) -> Any:  
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
    postorder_traversal(term, _enter_args, _exit_term)
    return args_stack[0][0]

def float_formatter(x: Value, *_) -> str:   
    if torch.is_tensor(x.value):
        return f"{x.value.item():.2f}"
    return f"{x.value:.2f}"

default_formatters = {
    Value: float_formatter,
    NonLeafStructure: lambda t, *args: f"(B{t.arity()} {' '.join(args)})",
    LeafStructure: lambda *_: "L"
}
    
def term_to_str(term: Term, formatters: dict = default_formatters) -> str: 
    ''' LISP style string '''
    def t_to_s(term: Term, args: list[str]):
        if term in formatters:
            return formatters[term](term, *args)
        term_type = type(term)
        if term_type in formatters:
            return formatters[term_type](term, *args)
        if isinstance(term, Wildcard) or isinstance(term, MetaVariable):
            return term.name
        if isinstance(term, Variable):
            return term.var_id
        if isinstance(term, Op):
            name = term.op_id
        else:
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
        size_cache = size_cache

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

def points_are_equiv(*ts: Term) -> bool:
    if len(ts) == 0:
        return True
    def rstrip(args: tuple[Term]):
        filtered = tuple(reversed([args[i] for i in range(len(args) - 1, -1, -1) if not (isinstance(args[i], Wildcard) and args[i].name == "*")]))
        return filtered
    arg_counts = [(len(sf), len(s) > 0 and (isinstance(s[-1], Wildcard) and s[-1].name == "*"))
                  for t in ts 
                  for s in [t.get_args()] 
                  for sf in [rstrip(s)]]
    max_count = max(ac for ac, _ in arg_counts)
    def are_same(term1: Term , term2: Term) -> bool:
        if type(term1) != type(term2):
            return False
        if isinstance(term1, Op):
            if term1.op_id != term2.op_id:
                return False
            return True 
        return term1 == term2  # assuming impl of _eq or ref eq     
    res = all(are_same(t, ts[0]) and (has_wildcard or (not has_wildcard and (ac == max_count))) for t, (ac, has_wildcard) in zip(ts, arg_counts))
    return res

def unify(b: UnifyBindings, *terms: Term, is_equiv: Callable = points_are_equiv) -> bool:
    ''' Unification of terms. Uppercase leaves are meta-variables, 
        ? is wildcard leaf - should not be used as operation
        * is wildcard args - 0 or more

        Note: we do not check here that bound meta-variables recursivelly resolve to concrete terms.
        This should be done by the caller.
    '''
    filtered_terms = [t for t in terms if not isinstance(t, Wildcard)]
    if len(filtered_terms) < 2:
        return True
    t_is_meta = [isinstance(t, MetaVariable) for t in filtered_terms]
    meta_operators = set([t.name for t, is_meta in zip(filtered_terms, t_is_meta) if is_meta])
    meta_terms = b.get(*meta_operators)
    bound_meta_terms = [bx for bx in meta_terms if bx is not None]
    concrete_terms = [t for t, is_meta in zip(filtered_terms, t_is_meta) if not is_meta]
    all_concrete_terms = bound_meta_terms + concrete_terms
    if len(all_concrete_terms) > 1:
        if not is_equiv(*all_concrete_terms):
            return False
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
        for arg_tuple in zip(*(t.get_args() for t in all_concrete_terms)):
            if not unify(b, *arg_tuple, is_equiv = is_equiv):
                return False
    return True

def match_term(term: Term, pattern: Term, is_equiv: Callable = points_are_equiv):
    ''' Search for all occurances of pattern in term. 
        * is wildcard leaf. X, Y, Z are meta-variables for non-linear matrching
    '''
    eq_terms = []
    def _exit_term(t: Term, *_):
        bindings = UnifyBindings()
        if unify(bindings, t, pattern, is_equiv = is_equiv):
            eq_terms.append((t, bindings))
        pass
    postorder_traversal(term, lambda *_: (), _exit_term)
    return eq_terms

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

def parse_term(term_str: str, i: int = 0, parsers = { Value: parse_float }) -> tuple[Term, int]:
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
    
# inf_count_constraints: dict[tuple, int] = defaultdict(lambda: math.inf)

# def signature_match(from_sig: tuple, to_sig: tuple) -> bool:
#     if len(from_sig) > len(to_sig):
#         return False
#     for x, y in zip(from_sig, to_sig):
#         if x != y:
#             return False
#     return True

# def check_signature_constraints(constraints: dict, signature: tuple) -> bool:
#     keys_to_check = [sig for sig in constraints.keys() if signature_match(sig, signature) ]
#     for i in range(len(signature)):
#         subsign = signature[:i+1]
#         if subsign in constraints and constraints[subsign] <= 0:
#             return False
#     return True

# error reasons of skeleton generation

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
            tape.resize((tape.shape[0] + buf_n, *tape[1:]))
            tape[tape.shape[0] - buf_n:] = alloc_tape()            
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
            print(str(new_tree))
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
                    print(str(new_tree))
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
                print(str(new_tree))
                return new_tree, next_pos_id
            print(f"\tfail {op_status}")
            if all(op_status == -1):
                return -1 
            elif all(op_status == 1):
                return 1
            return 0

    min_nonleaf_count = np.sum(min_counts) - min_counts[0] 
    skeleton, _ = _iter_rec(0, min_nonleaf_count, min_counts.copy(), max_counts.copy())

    return skeleton

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

    total_mins = np.cumsum(min_counts)
    total_maxs = np.cumsum(max_counts)
    min_totals = total - total_maxs + max_counts
    max_totals = total - total_mins + min_counts
    total_mins = total_mins[::-1]
    total_maxs = total_maxs[::-1]

    final_mins = np.maximum(total_mins, min_totals)
    final_maxs = np.minimum(total_maxs, max_totals)

    if np.any(final_maxs < final_mins):
        return None
    
    summed_counts = rnd.randint(final_mins, final_maxs + 1)

    sum_shifted = np.zeros_like(summed_counts)
    sum_shifted[:-1] = summed_counts[1:]
    counts = summed_counts - sum_shifted

    return counts

# test3 = _add_factorize(5, np.array([2, 1, 2]), np.array([3, 3, 3]))
# test1 = _add_factorize(10, np.array([1, 1, 0]), np.array([3, 3, 1]))
# test2 = _add_factorize(5, np.array([2, 1, 3]), np.array([3, 3, 3]))
# test4 = _add_factorize(5, np.array([0, 0, 0]), np.array([3, 3, 3]))
# pass 

def get_fn_arity(fn: Callable) -> int:
    signature = inspect.signature(fn)
    params = [p for p in signature.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_ONLY]
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
        return self._max_countss

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class BuilderAritites:
    arities: np.ndarray
    min_counts: np.ndarray
    max_counts: np.ndarray

UNBOUND = 1000000    

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class Builder:
    fn: Callable
    min_count: int = 0
    max_count: int = UNBOUND 

    def __post_init__(self):
        self._arity: int | None = None
        self.bound_id: int | None = None

    def arity(self) -> int:
        if self._arity is None:
            self._arity = get_fn_arity(self.fn)
        return self._arity

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
                b.bound_id = len(self._bound)
                self._bound.append(b)
            else:
                self._unbound.append(b)
        self._arity_builders: dict[int, ArityBuilders] | None = None
        self._arities: BuilderAritites | None = None
        self._min_counts: np.ndarray | None = None # only bound builders
        self._max_counts: np.ndarray | None = None # only bound builders

    def get_arity_builders(self) -> dict[int, ArityBuilders]:    
        if self._arity_builders is None:
            arity_builders = {}
            for b in self.builders:
                arity_builders.setdefault(b.arity(), []).append(b)

            self._arity_builders = {a: ArityBuilders(arity_builders[a]) for a in sorted(arity_builders.keys())}
            
            pass 
        return self._arity_builders
    
    def get_arities(self) -> BuilderAritites:
        ''' Returns total min and max counts for each arity '''
        if self._arities is None:
            arity_builders = self.get_arity_builders()
            arities = np.ndarray([a for a in arity_builders.keys()])
            min_counts = np.ndarray([np.sum(builders.get_min_counts()) for builders in arity_builders.values()])
            max_counts = np.ndarray([np.sum(builders.get_max_counts()) for builders in arity_builders.values()])
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
        new_builders = [b if b.bound_id is None else Builder(b.fn, min_counts[b.bound_id], max_counts[b.bound_id]) 
                        for b in self.builders ]
        builders = Builders(new_builders, self.get_builder_for_term)
        return builders

    def get_zero_counts(self) -> np.ndarray:
        return np.zeros(self._bound_count, dtype=int)


def instantiate_skeleton(skeleton: TermStructure, builders: Builders,
                    rnd: np.random.RandomState = np.random) -> Optional[Any]:
    # first we collect list of nodes for each arity 
    arity_terms = {}
    postorder_map(skeleton, lambda t,*_: arity_terms.setdefault(t.arity(), []).append(t),
                  with_cache=False)
    arity_builders = {}
    for arity, arity_builders in builders.get_arity_builders().items():
        terms = arity_terms.get(arity, [])
        # should we check arity count sum() of mins and maxes?? 
        split = _add_factorize(len(terms),
                               min_counts=arity_builders.get_min_counts(),
                               max_counts=arity_builders.get_max_counts(), rnd = rnd)
        if split is None:
            return None
        builders = [b_inst for bi, b in enumerate(arity_builders.builders)
                    for b_inst in [b] * split[bi]]
        rnd.shuffle(builders)
        arity_builders[arity] = builders

    occurs = {}

    def _init_term(term: TermStructure, *args):
        arity = term.arity()
        cur_occur = occurs.setdefault(arity, 0)
        builder = arity_builders[arity][cur_occur]
        term = builder.fn(*args)
        occurs[arity] += 1
        return term 
    
    instance = postorder_map(skeleton, _init_term, with_cache=False)

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
    
    if map_skeleton is None:
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

def select_positions(positions: list[TermPos], num_positions: int,
                        select_node_leaf_prob: Optional[float] = 0.1,
                        rnd: np.random.RandomState = np.random) -> list[int]:
    # selecting poss for given number of mutants 
    pos_proba = get_pos_scores(positions, select_node_leaf_prob = select_node_leaf_prob, rnd = rnd)
    ordered_pos_ids = np.argsort(pos_proba).tolist()
    if len(ordered_pos_ids) > num_positions:
        ordered_pos_ids = ordered_pos_ids[:num_positions]
    elif len(ordered_pos_ids) < num_positions:
        repeat_cnt = math.ceil(num_positions / len(ordered_pos_ids))
        ordered_pos_ids = (ordered_pos_ids * repeat_cnt)[:num_positions]
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

    selected_pos = select_positions(term, pos_list, num_children, select_node_leaf_prob = select_node_leaf_prob, rnd = rnd)

    mutants = []
    for pos_id in selected_pos:
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
                            grow_leaf_prob = None, rnd = rnd)
        if new_child is None:
            mutants.append(term) # noop
        else:
            mutated_term = replace(builders, position, new_child, positions)
            # child_depth = get_depth(mutated_term)
            # assert child_depth <= tree_max_depth
            mutants.append(mutated_term)
        
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

    if len(positions1) == 0:
        pos_list1 = [TermPos(term1)]
    else:
        pos_list1 = list(positions1.keys())
    pos_proba1 = get_pos_scores(term1, pos_list1,
                                select_node_leaf_prob = select_node_leaf_prob, 
                                rnd = rnd)    
    pos_ids1 = np.argsort(pos_proba1)
            
    if len(positions2) == 0:
        pos_list2 = [TermPos(term2)]
    else:
        pos_list2 = list(positions2.keys())
    pos_proba2 = get_pos_scores(term2, pos_list2,
                                select_node_leaf_prob = select_node_leaf_prob, 
                                rnd = rnd)
    
    pos_ids2 = np.argsort(pos_proba2)
    
    selected_pairs = []

    while len(selected_pairs) < num_children:
        for pos_id1, pos_id2 in product(pos_ids1, pos_ids2):
            pos1 = pos_list1[pos_id1]
            pos2 = pos_list2[pos_id2]
            if (pos1.at_depth + get_depth(pos2.term, depth_cache) <= tree_max_depth) and \
                replacement_counts_sat(pos1, pos2.term, term1, builders, count_cache):
                selected_pairs.append((pos1, positions1, pos2.term))
                if len(selected_pairs) >= num_children:
                    break
            if (pos2.at_depth + get_depth(pos1.term, depth_cache) <= tree_max_depth) and \
                replacement_counts_sat(pos2, pos1.term, term2, builders, count_cache):
                selected_pairs.append((pos2, positions2, pos1.term))
                if len(selected_pairs) >= num_children:
                    break

    children = []
    for pos, poss, term in selected_pairs:
        new_child = replace(builders, pos, term, poss)
        # child_depth = get_depth(new_child, depth_cache)
        # assert child_depth <= tree_max_depth
        children.append(new_child)

    return children

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
    # t1, _ = parse_term("(f x y)")
    p1, _ = parse_term("(f (f X X) Y Y)")
    # p1, _ = parse_term("(f *)")
    matches = match_term(t1, p1)
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