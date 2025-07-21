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

from collections import defaultdict, deque
from dataclasses import dataclass, field
import inspect
from itertools import product
import math
from typing import Any, Callable, Generator, Optional, Sequence, Type

import numpy as np


@dataclass(frozen=True)
class TermSignature:
    ''' Descriptive, have to be stateless - state is outside'''
    name: str
    arity: int = 0
    category: str = "" # type name or other grouping
    
@dataclass(eq=False, unsafe_hash=False)
class Term:
    signature: TermSignature
    args: list['Term'] = field(default_factory=list)
    # NOTE: bindings are not detouched from term 
    # bindings: dict[int, int] = field(default_factory=dict)
    # ''' Binding defines index in semantics, the vector which specifies evaluation of the term 
    #     Map: evaluation id to semantics id 
    # '''


# @dataclass
# class TermPos:
#     ''' Index of position by Term object and its occurance
#         We avoid adding other data to minimize memory footprint.
#     '''
#     child_term: Term 
#     occur: int 

#     # idx: int # global index it postorder
#     # depth: int 

# @dataclass(eq=False, unsafe_hash=False, kw_only=True)
# class BindingValue: 
#     value: Optional[Any] = None
#     value_builder: Optional[Callable] = None

def sign_from_fn(fn: Callable, name: Optional[str] = None) -> TermSignature:
    ''' Get signature from function '''
    s = inspect.signature(fn)
    arity = len(s.parameters)
    name = name or fn.__name__
    category = name # TODO: get from function signature
    return TermSignature(name=name, arity=arity, category=category)

def build_term(term_cache: dict[tuple, Term], signature: str | TermSignature, args: Sequence[Term] = [], 
               cache_cb: Callable = lambda t,hit:()) -> Term:
    if type(signature) is str:
        signature = TermSignature(signature, arity = len(args), category="")
    key = (signature, *args)
    if key not in term_cache:
        term = Term(signature, list(args))
        cache_cb(term, False)
        term_cache[key] = term
    else:
        term = term_cache[key]
        cache_cb(term, True)
    return term

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
        if cur_arg_i >= len(cur_term.args):
            status = exit_term(cur_term, cur_term_i, cur_parent)
            if status == TRAVERSAL_EXIT:
                return
        else:
            cur_arg = cur_term.args[cur_arg_i]
            q.appendleft((cur_arg_i + 1, cur_term, cur_term_i, cur_parent))
            q.appendleft((0, cur_arg, cur_arg_i, cur_term))


def postorder_map(term: Term, fn: Callable, with_cache = False) -> Any:  
    args_stack = [[]]
    term_cache = {}
    if with_cache:
        def add_res(t: Term, res: Any):
            term_cache[t] = res
    else:
        def add_res(t: Term, res: Any):
            pass
    def _enter_args(t: Term, term_i, p: Term):
        if t in term_cache:
            processed_t = term_cache[t]
            args_stack[-1].append(processed_t)
            return TRAVERSAL_EXIT_NODE
        args_stack.append([])
    def _exit_term(t: Term, term_i, p: Term):
        term_processed_args = args_stack.pop()
        processed_t = fn(t, term_processed_args)
        add_res(t, processed_t)
        args_stack[-1].append(processed_t) #add to parent args
    postorder_traversal(term, _enter_args, _exit_term)
    return args_stack[0][0]

def term_to_str(term: Term) -> str: 
    ''' LISP style string '''
    def t_to_s(term: Term, args: list[str]):
        if len(args) == 0:
            return term.signature.name
        return "(" + " ".join([term.signature.name, *args]) + ")"    
    res = postorder_map(term, t_to_s, with_cache=True)
    return res 

Term.__repr__ = term_to_str
Term.__str__ = term_to_str

def get_leaves(root: Term, name: Optional[str | TermSignature] = None, leaves_cache = {}) -> list[Term]:
    ''' Find all leaves in root that are equal to term by name '''
    if root not in leaves_cache:
        leaves = []
        def _exit_term(term: Term, *_):
            if len(term.args) == 0:
                leaves.append(term)
        postorder_traversal(root, lambda *_: (), _exit_term)
        leaves_cache[root] = leaves
    leaves = leaves_cache[root]
    if name is None:
        return leaves
    filtered_leaves = [child_term for child_term in leaves if child_term.signature.name == name or child_term.signature == name]
    return filtered_leaves

@dataclass(eq=False, unsafe_hash=False)
class TermPos:
    term: Term
    occur: int = 0 # -> id of term occrance in root
    pos: int = 0 # pos in parent args
    depth: int = 0
    at_depth: int = 0

    def __eq__(self, other):
        if isinstance(other, TermPos):
            return self.term == other.term and self.occur == other.occur
        return False

    def __hash__(self):
        return hash((self.term, self.occur))    

def get_depths(term: Term, depth_cache: Optional[dict[Term, int]] = None) -> dict[Term, int]:
    
    depth_cache = depth_cache or {}

    def update_parent(term: Term, parent: Term):
        if parent is not None: 
            depth_cache[parent] = max(depth_cache.get(parent, -1), depth_cache.get(term, 0))        

    def _exit_term(term: Term, term_i, parent: Term):
        depth_cache[term] = depth_cache.get(term, -1) + 1
        update_parent(term, parent)

    def _enter_args(term: Term, term_i, parent: Term):
        if term in depth_cache:
            update_parent(term, parent)
            return TRAVERSAL_EXIT_NODE # skip args
    

    postorder_traversal(term, _enter_args, _exit_term) 

    return depth_cache   

def get_term_pos(term: Term, depth_cache: dict[Term, int]) -> dict[TermPos, TermPos]: 
    ''' Returns dictionary where keys are all positions in the term and values are references to parent position 
        NOTE: we do not return thee root of the term as TermPos as it does not have parent
    '''

    get_depths(term, depth_cache)
    
    subterms: dict[TermPos, TermPos] = {}
    term_at_depths = {}
    last_term_pos: dict[Term, TermPos] = {}
    term_occur = {}
    def _enter_args(term: Term, term_i, parent: Term):
        cur_occur = term_occur.get(term, 0)
        term_occur[term] = cur_occur + 1
        term_at_depths[term] = term_at_depths.get(parent, -1) + 1
        term_pos = TermPos(term, cur_occur, term_i, depth_cache.get(term, 0), term_at_depths.get(term, 0))
        last_term_pos[term] = term_pos
        parent_pos = last_term_pos.get(parent, None)
        if parent_pos is not None:
            subterms[term_pos] = parent_pos

    def _exit_term(term: Term, *_):
        del last_term_pos[term]
        del term_at_depths[term]

    postorder_traversal(term, _enter_args, _exit_term)

    return subterms

def pick_term_pos(term: Term, depth_cache: dict[Term, int], 
                   pred: Callable[[TermPos], tuple[bool, bool]]) -> dict[TermPos, TermPos]:
    ''' Return TermPos that satisfy given predicate. Allows early termination (find_first patern)
        NOTE: we do not return thee root of the term as TermPos as it does not have parent
    '''
    get_depths(term, depth_cache)
    
    selected_pos = []
    subterms: dict[TermPos, TermPos] = {}
    term_at_depths = {}
    last_term_pos: dict[Term, TermPos] = {}
    term_occur = {}
    def _exit_term(term: Term, term_i, parent: Term):
        cur_occur = term_occur.get(term, 0)
        term_occur[term] = cur_occur + 1
        term_at_depths[term] = term_at_depths.get(parent, -1) + 1
        term_pos = TermPos(term, cur_occur, term_i, depth_cache.get(term, 0), term_at_depths.get(term, 0))
        last_term_pos[term] = term_pos
        should_pick, should_break = pred(term_pos)
        parent_pos = last_term_pos.get(parent, None)
        if parent_pos is not None:
            subterms[term_pos] = parent_pos
        if should_pick:
            selected_pos.append(term_pos)
        if should_break:
            return TRAVERSAL_EXIT

    postorder_traversal(term, lambda *_:(), _exit_term)

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
    arg_counts = [(len(sf), len(s) > 0 and s[-1] == "*") for t in ts for s in ["".join(a.signature.name for a in t.args)] for sf in [s.rstrip("*")]]
    max_count = max(ac for ac, _ in arg_counts)
    res = all(t.signature.name == ts[0].signature.name and (has_wildcard or (not has_wildcard and (ac == max_count))) for t, (ac, has_wildcard) in zip(ts, arg_counts))
    return res

def unify(b: UnifyBindings, is_equiv: Callable, *terms: Term) -> bool:
    ''' Unification of terms. Uppercase leaves are meta-variables, 
        ? is wildcard leaf - should not be used as operation
        * is wildcard args - 0 or more

        Note: we do not check here that bound meta-variables recursivelly resolve to concrete terms.
        This should be done by the caller.
    '''
    filtered_terms = [t for t in terms if t.signature.name not in "?*"]
    if len(filtered_terms) < 2:
        return True
    t_is_meta = [t.signature.name.isupper() for t in filtered_terms]
    meta_operators = set([t.signature.name for t, is_meta in zip(filtered_terms, t_is_meta) if is_meta])
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
        for arg_tuple in zip(*(t.args for t in all_concrete_terms)):
            if not unify(b, is_equiv, *arg_tuple):
                return False
    return True

def match_term(term: Term, pattern: Term, is_equiv: Callable = points_are_equiv):
    ''' Search for all occurances of pattern in term. 
        * is wildcard leaf. X, Y, Z are meta-variables for non-linear matrching
    '''
    eq_terms = []
    def _exit_term(t: Term, term_i: int, p: Term):
        bindings = UnifyBindings()
        if unify(bindings, is_equiv, t, pattern):
            eq_terms.append((t, bindings))
        pass
    postorder_traversal(term, lambda *_: (), _exit_term)
    return eq_terms

def bind_terms(terms: list[Term], values: Any | list[Optional[Any]]) -> dict[tuple[Term, int], Any]:
    ''' Manually asign semantic vector id to a specific term in specific evaluation '''    
    if len(terms) == 0:
        return 
    if type(values) is not list:
        values = [values] * len(terms)
    values += [None] * (len(terms) - len(values))
    res = {}
    for leaf_id, (leaf, value) in enumerate(zip(terms, values)):
        res[(leaf, leaf_id)] = value
    return res # None for unset bindings

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

def parse_term(term_cache, term_str: str, i: int = 0) -> tuple[Term, int]:
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
                if type(arg) is Term:
                    args.append(arg)
                elif type(arg) is tuple: 
                    bindings[arg[0]] = arg[1]
            term = build_term(term_cache, name, args)
            branches[0].append(term)
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
            leaf = build_term(term_cache, literal)
            branches[0].append(leaf)
    return branches[0][0], i

def replace(term_builder, term_pos: TermPos, with_term: Term, term_parents: dict[TermPos, TermPos]) -> Term:
    cur_pos = term_pos
    new_term = with_term
    while cur_pos in term_parents:

        cur_parent = term_parents[cur_pos]

        new_parent_term_args = [*cur_parent.term.args[:cur_pos.pos], new_term, *cur_parent.term.args[cur_pos.pos + 1:]]
        new_term = term_builder(cur_parent.term.signature, new_parent_term_args)

        cur_pos = cur_parent
        
    return new_term

def evaluate(term: Term, ops: dict[str | TermSignature, Callable],
                get_binding: Callable[[tuple[Term, int]], Any] = lambda ti: None,
                set_binding: Callable[[tuple[Term, int], Any], Any] = lambda ti,v:()) -> Any:
    ''' Fully or partially evaluates term (concrete or abstract) '''
    
    term_occur = {}
    args_stack = [[]]
    def _enter_args(term: Term, term_i, parent: Term):
        cur_occur = term_occur.get(term, 0)        
        res = get_binding((term, cur_occur))
        if res is not None:
            args_stack[-1].append(res)
            return TRAVERSAL_EXIT_NODE
        args_stack.append([])
        
    def _exit_term(term: Term, term_i, p: Term):
        cur_occur = term_occur.get(term, 0)
        term_occur[term] = cur_occur + 1
        args = args_stack.pop()
        if all(arg is not None for arg in args):
            op_fn = ops.get(term.signature, None) or ops.get(term.signature.name, None)
            if op_fn is not None:
                res = op_fn(*args)
        if res is not None:            
            set_binding((term, cur_occur), res)
        args_stack[-1].append(res)

    postorder_traversal(term, _enter_args, _exit_term)

    return args_stack[0][0]
    
inf_count_constraints: dict[str, int] = defaultdict(lambda: math.inf)

def grow(term_builder: Callable, leaf_ops: list[TermSignature], branch_ops: list[TermSignature],
         grow_depth = 5, grow_leaf_prob: Optional[float] = None,
         counts_constraints: dict[str, int] = inf_count_constraints,
         rnd: np.random.RandomState = np.random) -> Optional[Term]:
    ''' Grow a tree with a given depth '''
    allowed_leaves = [t for t in leaf_ops if counts_constraints[t.category] > 0]
    allowed_branches = [t for t in branch_ops if counts_constraints[t.category] > 0]
    args = []
    if (grow_depth == 0) or len(allowed_branches) == 0:
        if len(allowed_leaves) == 0:
            return None 
        leaf_index = rnd.choice(len(allowed_leaves))
        selected_op = allowed_leaves[leaf_index]
        counts_constraints[selected_op.category] -= 1        
    else:
        disallowed_category = set() #already attempted with None results
        while True: # backtrack in case if counts_constraints were noto satisfied - attempting other symbol - or return none
            iter_allowed_leaves = [t for t in allowed_leaves if t.category not in disallowed_category]
            iter_allowed_branches = [f for f in allowed_branches if f.category not in disallowed_category]
            if len(iter_allowed_branches) == 0 and len(iter_allowed_leaves) == 0:
                return None
            if grow_leaf_prob is None:
                func_index = rnd.choice(len(iter_allowed_branches) + len(iter_allowed_leaves))
                selected_op = iter_allowed_branches[func_index] if func_index < len(iter_allowed_branches) else iter_allowed_leaves[func_index - len(iter_allowed_branches)]
            elif len(iter_allowed_leaves) > 0 and rnd.rand() < grow_leaf_prob:
                leaf_index = rnd.choice(len(iter_allowed_leaves))
                selected_op = iter_allowed_leaves[leaf_index]
            else:
                func_index = rnd.choice(len(iter_allowed_branches))
                selected_op = iter_allowed_branches[func_index]
            new_counts_constraints = dict(counts_constraints)
            new_counts_constraints[selected_op.category] -= 1            
            for _ in range(selected_op.arity):
                term = grow(term_builder, allowed_leaves, allowed_branches,
                            grow_depth = grow_depth - 1, grow_leaf_prob = grow_leaf_prob, 
                            counts_constraints = new_counts_constraints, rnd = rnd)
                args.append(term)
                if term is None:
                    break
            if len(args) > 0 and args[-1] is None:
                disallowed_category.add(selected_op)
                continue 
            counts_constraints.update(new_counts_constraints)
            break
    return term_builder(selected_op, args)

def full(term_builder: Callable, leaf_ops: list[TermSignature], branch_ops: list[TermSignature],
         full_depth = 5,
         counts_constraints: dict[str, int] = inf_count_constraints,
         rnd: np.random.RandomState = np.random) -> Optional[Term]:
    ''' Grow a tree with a given depth '''
    allowed_leaves = [t for t in leaf_ops if counts_constraints[t.category] > 0]    
    allowed_branches = [f for f in branch_ops if counts_constraints[f.category] > 0]
    args = []
    if full_depth == 0 or len(allowed_branches) == 0:
        if len(allowed_leaves) == 0:
            return None         
        leaf_id = rnd.choice(len(allowed_leaves))
        selected_op = allowed_leaves[leaf_id]
        counts_constraints[selected_op.category] -= 1          
    else:
        disallowed_category = set() #already attempted with None results
        while True: # backtrack in case if counts_constraints were noto satisfied - attempting other symbol - or return none
            iter_allowed_branches = [f for f in allowed_branches if f.category not in disallowed_category]
            if len(iter_allowed_branches) == 0:
                return None
            branch_id = rnd.choice(len(iter_allowed_branches))
            selected_op = iter_allowed_branches[branch_id]
            new_counts_constraints = dict(counts_constraints)
            new_counts_constraints[selected_op.category] -= 1
            for _ in range(selected_op.arity):
                node = full(term_builder, allowed_leaves, allowed_branches, 
                            full_depth=full_depth - 1, counts_constraints = new_counts_constraints, 
                            rnd = rnd)
                args.append(node)
                if node is None:
                    break 
            if len(args) > 0 and args[-1] is None:
                disallowed_category.add(selected_op.category)
                continue  
            if counts_constraints is not None:
                counts_constraints.update(new_counts_constraints)       
            break  
    return term_builder(selected_op, args)


def ramped_half_and_half(term_builder: Callable, leaf_ops: list[TermSignature], branch_ops: list[TermSignature],
                         rhh_min_depth = 1, rhh_max_depth = 5, rhh_grow_prob = 0.5, 
                         counts_constraints: dict[str, int] = inf_count_constraints,                
                         rnd: np.random.RandomState = np.random) -> Optional[Term]:
    ''' Generate a population of half full and half grow trees '''
    depth = rnd.randint(rhh_min_depth, rhh_max_depth+1)
    if rnd.rand() < rhh_grow_prob:
        return grow(term_builder, leaf_ops, branch_ops, grow_depth = depth, counts_constraints = counts_constraints, rnd = rnd)
    else:
        return full(term_builder, leaf_ops, branch_ops, full_depth = depth, counts_constraints = counts_constraints, rnd = rnd)
   

# IDEA: dropout in GP, frozen tree positions which cannot be mutated or crossovered - for later
# def select_rnd_pos(term: Term, positions: Sequence[TermPos],
#                                 select_node_leaf_prob: Optional[float] = 0.1,
#                                 rnd: np.random.RandomState = np.random,
#                                 exclude_root = True) -> Optional[TermPos]:
#     if len(positions) == 0:
#         return None
#     if select_node_leaf_prob is None:
#         selected_id = rnd.randint(1, len(positions)) # excluding root
#         selected_pos = positions[selected_id]
#         return selected_pos
#     nonleaves = []
#     leaves = [] 
#     for child_id in range(len(positions)):
#         child = positions[child_id]
#         if len(child.term.args) > 0:
#             nonleaves.append(child)
#         else:
#             leaves.append(child) 
#     if len(nonleaves) == 0 and len(leaves) == 0:
#         return None
#     if (rnd.rand() < select_node_leaf_prob and len(leaves) > 0) or len(nonleaves) == 0:
#         selected_idx = rnd.choice(len(leaves))
#         selected_pos = leaves[selected_idx]
#     else:
#         selected_idx = rnd.choice(len(nonleaves))
#         selected_pos = nonleaves[selected_idx]
#     return selected_pos

def get_pos_scores(term: Term, positions: list[TermPos],
                        select_node_leaf_prob: Optional[float] = 0.1,
                        rnd: np.random.RandomState = np.random) -> Optional[np.ndarray]:
    pos_proba = rnd.rand(len(positions))
    if select_node_leaf_prob is not None:
        proba_mod = np.array([select_node_leaf_prob if len(pos.term.args) == 0 else (1 - select_node_leaf_prob) for pos in positions ])
        pos_proba *= proba_mod
    return pos_proba

def select_positions(term: Term, positions: list[TermPos], num_positions: int,
                        select_node_leaf_prob: Optional[float] = 0.1,
                        rnd: np.random.RandomState = np.random) -> list[int]:
    # selecting poss for given number of mutants 
    pos_proba = get_pos_scores(term, positions, select_node_leaf_prob = select_node_leaf_prob, rnd = rnd)
    if pos_proba is None:
        return []
    ordered_pos_ids = np.argsort(pos_proba)[-1:-num_positions-1:-1].tolist()
    if len(ordered_pos_ids) < num_positions:
        repeat_cnt = math.ceil(num_positions / len(ordered_pos_ids))
        ordered_pos_ids = (ordered_pos_ids * repeat_cnt)[:num_positions]
    return ordered_pos_ids


def one_point_rand_mutation(term_builder, term: Term, positions: dict[TermPos, TermPos], 
                            leaf_ops: list[TermSignature], branch_ops: list[TermSignature],
                            rnd: np.random.RandomState = np.random,
                            select_node_leaf_prob: Optional[float] = 0.1,
                            # include_root = True, 
                            tree_max_depth = 17,
                            num_children = 1) -> list[Term]:
    # if include_root:
    pos_list = [TermPos(term), *positions.keys()] # we always include root to avoid checks after selection
    # else:
    #     pos_list = list(positions.keys())
    selected_pos = select_positions(term, pos_list, num_children, select_node_leaf_prob = select_node_leaf_prob, rnd = rnd)
    # if len(selected_pos) == 0:
    #     return [term] * num_children # noop
    mutants = []
    for pos_id in selected_pos:
        position: TermPos = pos_list[pos_id]
        new_child = grow(term_builder, leaf_ops, branch_ops, grow_depth = min(5, tree_max_depth - position.at_depth), 
                            grow_leaf_prob = None, rnd = rnd)
        if new_child is None:
            mutants.append(term) # noop
        else:
            mutated_term = replace(term_builder, position, new_child, positions)
            mutants.append(mutated_term)
        
    return mutants

def one_point_rand_crossover(term_builder, term1: Term, term2: Term,
                                           positions1: dict[TermPos, TermPos], positions2: dict[TermPos, TermPos],
                                rnd: np.random.RandomState = np.random,
                                select_node_leaf_prob: Optional[float] = 0.1,
                                include_root = True, tree_max_depth = 17,
                                num_children = 1):
    pos_list1 = [TermPos(term1), *positions1.keys() ]
    pos_proba1 = get_pos_scores(term1, pos_list1, include_root = include_root,
                                select_node_leaf_prob = select_node_leaf_prob, rnd = rnd)
    
    pos_ids1 = np.argsort(pos_proba1)
            
    pos_list2 = [TermPos(term2), *positions2.keys()]
    pos_proba2 = get_pos_scores(term2, pos_list2, include_root = include_root,
                                select_node_leaf_prob = select_node_leaf_prob, rnd = rnd)
    
    pos_ids2 = np.argsort(pos_proba2)
    
    selected_pairs = []

    while len(selected_pairs) < num_children:
        for pos_id1, pos_id2 in product(reversed(pos_ids1), reversed(pos_ids2)):
            pos1 = pos_list1[pos_id1]
            pos2 = pos_list2[pos_id2]
            if pos1.at_depth + pos2.depth <= tree_max_depth:
                selected_pairs.append((pos1, positions1, pos2.term))
            if len(selected_pairs) >= num_children:
                break
            if pos2.at_depth + pos1.depth <= tree_max_depth:
                selected_pairs.append((pos2, positions2, pos1.term))
            if len(selected_pairs) >= num_children:
                break

    children = []
    for pos, poss, term in selected_pairs:
        new_child = replace(term_builder, pos, term, poss)
        children.append(new_child)

    return children


if __name__ == "__main__":

    term_cache = {}
    # tests
    # t1, _ = parse_term(term_cache, "(f (f X (f x (f x)) (f x (f x))))")
    # t2, _ = parse_term(term_cache, "(f (f (f x x) Y Y))")
    # t3, _ = parse_term(term_cache, "(f Z)")
    # b = UnifyBindings()
    # res = unify(b, points_are_equiv, t1, t2, t3)
    pass


    t1_str = "(f (f (f x x) (f x (f x)) (f x (f x))))"
    t1, _ = parse_term(term_cache, t1_str)

    depth = get_depths(t1)
    print(depth)
    pass    

    print(term_to_str(t1))
    assert term_to_str(t1) == t1_str, f"Expected {t1_str}, got {term_to_str(t1)}"
    pass
    # t1, _ = parse_term("(f x y z)")
    # p1, _ = parse_term(term_cache, "(f (f X X) Y Y)")
    p1, _ = parse_term(term_cache, "(f ? ? *)")
    matches = match_term(t1, p1)
    matches = [(term_to_str(m[0]), {k:term_to_str(v) for k, v in m[1].bindings.items()}) for m in matches]
    pass


    # res, _ = parse_term("  \n(   f   (g    x :0:1)  (h \nx) :0:12)  \n", 0)
    t1, _ = parse_term(term_cache, "  \n(   f   (g    x)  (h \nx))  \n", 0)
    leaves = get_leaves(t1, "x", leaves_cache = {})
    bindings = bind_terms(leaves, 1)
    print(term_to_str(t1))
    ev1 = evaluate(t1, {"f": lambda x, y: x + y, "g": lambda x: x * 2, "h": lambda x: x ** 2}, bindings.get, bindings.setdefault)

    pass    