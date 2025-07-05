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
import math
from typing import Any, Callable, Generator, Optional, Sequence, Type

import numpy as np

@dataclass(frozen=True)
class TermType: 
    arity: int = 0
    category: str = ""

zero_type = TermType()    

@dataclass(frozen=True)
class TermSignature:
    ''' Descriptive, have to be stateless - state is outside'''
    name: str
    type: TermType = zero_type
    
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

class TermBindings:
    ''' One term invocation. Includes eager or lazy binding of term or term at position
    '''

    def __init__(self,
                       term_bindings: dict[tuple[Term, int] | Term, Any | Callable] = {},
                       op_bindings: dict[tuple[TermSignature, int] | TermSignature, Any | Callable] = {},
                       frozen = False):
        self.result: Any = None # root binding
        self.term_bindings = term_bindings
        self.op_bindings = op_bindings
        self.only_term_binds = all(k for k in term_bindings if type(k) is Term)
        self.frozen = frozen
        self.hits = 0
        self.misses = 0

    def get(self, term: Term, term_pos: int):
        res = self.term_bindings.get((term, term_pos), None) or self.term_bindings.get(term, None) or self.op_bindings.get((term.signature, term_pos), None) or self.op_bindings.get(term.signature, None)
        if res is None:
            self.misses += 1
        elif not self.frozen:
            self.hits += 1
        return res
    
    def set(self, term: Term, term_pos: int, value: Any):
        if self.frozen:
            return 
        binding_key = (term, term_pos) if self.only_term_binds else term
        self.term_bindings[binding_key] = value        

def build_term(term_cache: dict[tuple, Term], signature: str | TermSignature, args: Sequence[Term] = [], 
               cache_cb: Callable = lambda t,hit:()) -> Term:
    if type(signature) is str:
        signature = TermSignature(signature, TermType(arity = len(args), category=signature))
    key = (signature, *args)
    if key not in term_cache:
        term = Term(signature, list(args))
        cache_cb(term, False)
        term_cache[key] = term
    else:
        term = term_cache[key]
        cache_cb(term, True)
    return term
    
def postorder_traversal(term: Term, enter_args: Callable, exit_term: Callable):
    q = deque([(0, term, None)])
    while len(q) > 0:
        cur_i, cur_term, cur_parent = q.popleft()
        if cur_i == 0:
            enter_args(cur_term, cur_parent)
        if cur_i >= len(cur_term.args):
            exit_term(cur_term, cur_parent)
        else:
            cur_arg = cur_term.args[cur_i]
            q.appendleft((cur_i + 1, cur_term, cur_parent))
            q.appendleft((0, cur_arg, cur_term))

def postorder_map(term: Term, fn: Callable) -> Any:    
    term_args = {None: []}    
    def _enter_args(t: Term, p: Term):
        term_args[t] = []
    def _exit_term(t: Term, p: Term):
        processed_t = fn(t, term_args[t])
        term_args[p].append(processed_t)
        del term_args[t]
    postorder_traversal(term, _enter_args, _exit_term)
    return term_args[None][0]

def term_to_str(term: Term) -> str: 
    ''' LISP style string '''
    def t_to_s(term: Term, args: list[str]):
        if len(args) == 0:
            return term.signature.name
        return "(" + " ".join([term.signature.name, *args]) + ")"    
    res = postorder_map(term, t_to_s)
    return res 

def get_leaves(root: Term, name: Optional[str | TermSignature] = None, leaves_cache = {}) -> list[Term]:
    ''' Find all leaves in root that are equal to term by name '''
    if root not in leaves_cache:
        leaves = []
        def _exit_term(term: Term, parent: Term):
            if len(term.args) == 0:
                leaves.append(term)
        postorder_traversal(root, lambda t,p: (), _exit_term)
        leaves_cache[root] = leaves
    leaves = leaves_cache[root]
    if name is None:
        return leaves
    filtered_leaves = [child_term for child_term in leaves if child_term.signature.name == name or child_term.signature == name]
    return filtered_leaves

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
    def _exit_term(t: Term, p: Term):
        bindings = UnifyBindings()
        if unify(bindings, is_equiv, t, pattern):
            eq_terms.append((t, bindings))
        pass
    postorder_traversal(term, lambda t,p: (), _exit_term)
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

# parents should be eventually cached
def get_parents(term: Term):
    term_poss = {}
    term_parents = {}
    def _enter(term: Term, parent: Optional[Term]):
        term_pos = term_poss.get(term, -1) + 1
        term_poss[term] = term_pos
        if parent in term_poss:
            parent_pos = term_poss[parent] - 1 # parent term cannot be inside itself
            term_parents[(term, term_pos)] = (parent, parent_pos)
    postorder_traversal(term, _enter, lambda t,p: ())
    return term_parents

def get_chain_to_root(term_pos: tuple[Term, int], term_parents: dict[tuple[Term, int], tuple[Term, int]]) -> Generator[tuple[Term, int], None, None]:
    ''' Get chain of term positions to root '''
    yield term_pos
    cur_pos = term_pos
    while cur_pos in term_parents:
        cur_pos = term_parents[cur_pos]
        yield cur_pos

def evaluate(term: Term, ops: dict[str | TermSignature, Callable], ev: TermBindings) -> Any:
    ''' Fully or partially evaluates term (concrete or abstract) '''
    term_poss = {}
 
    def _eval(term: Term, args: list[Any]):
        term_pos = term_poss.get(term, -1) + 1
        term_poss[term] = term_pos
        if any(arg is None for arg in args):
            return None        
        binding = ev.get(term, term_pos)
        res = None 
        if binding is not None:
            if callable(binding):
                res = binding(*args)
            else: 
                return binding
        if res is None:
            op_fn = ops.get(term.signature, None) or ops.get(term.signature.name, None)
            if op_fn is not None:
                res = op_fn(*args)
        ev.set(term, term_pos, res)
        return res
    res_semantics = postorder_map(term, _eval)
    ev.result = res_semantics
    return res_semantics

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
   
if __name__ == "__main__":

    term_cache = {}
    # tests
    # t1, _ = parse_term(term_cache, "(f (f X (f x (f x)) (f x (f x))))")
    # t2, _ = parse_term(term_cache, "(f (f (f x x) Y Y))")
    # t3, _ = parse_term(term_cache, "(f Z)")
    # b = UnifyBindings()
    # res = unify(b, points_are_equiv, t1, t2, t3)
    pass


    t1, _ = parse_term(term_cache, "(f (f (f x x) (f x (f x)) (f x (f x))))")
    # t1, _ = parse_term("(f x y z)")
    # p1, _ = parse_term(term_cache, "(f (f X X) Y Y)")
    p1, _ = parse_term(term_cache, "(f ? ? *)")
    matches = match_term(t1, p1)
    matches = [(term_to_str(m[0]), {k:term_to_str(v) for k, v in m[1].bindings.items()}) for m in matches]
    pass


    # res, _ = parse_term("  \n(   f   (g    x :0:1)  (h \nx) :0:12)  \n", 0)
    t1, _ = parse_term(term_cache, "  \n(   f   (g    x)  (h \nx))  \n", 0)
    leaves = get_leaves(t1, "x", leaves_cache = {})
    evd1 = TermBindings(bind_terms(leaves, 1))
    print(term_to_str(t1))
    ev1 = evaluate(t1, {"f": lambda x, y: x + y, "g": lambda x: x * 2, "h": lambda x: x ** 2}, evd1)
    pass    