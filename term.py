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
from typing import Any, Callable, Generator, Optional, Type

from semantics import DummySemanticIndex, SemanticIndex

@dataclass(eq=False, unsafe_hash=False)
class Term:
    operator: str
    args: list['Term'] = field(default_factory=list)
    bindings: dict[int, int] = field(default_factory=dict)
    ''' Binding defines index in semantics, the vector which specifies evaluation of the term 
        Map: evaluation id to semantics id 
    '''
    
def postorder_traversal(term: Term, enter_args: Callable, exit_term: Callable):
    q = deque([(0, term, None)])
    while len(q) > 0:
        cur_i, cur_term, cur_parent = q.popleft()
        if cur_i == 0:
            enter_args(cur_term)
        if cur_i >= len(cur_term.args):
            exit_term(cur_term, cur_parent)
        else:
            cur_arg = cur_term.args[cur_i]
            q.appendleft((cur_i + 1, cur_term, cur_parent))
            q.appendleft((0, cur_arg, cur_term))

def postorder_map(term: Term, fn: Callable) -> Any:    
    term_args = {None: []}    
    def _enter_args(t: Term):
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
        binds = [f":{eid}:{sid}" for eid, sid in term.bindings.items()]
        if len(args) == 0 and len(binds) == 0:
            return term.operator
        return "(" + " ".join([term.operator, *args, *binds]) + ")"    
    res = postorder_map(term, t_to_s)
    return res 

def get_leaves(root: Term, operator: Optional[str] = None, leaves_cache = {}) -> list[Term]:
    ''' Find all leaves in root that are equal to term by name (ignoring bindings) '''
    if root not in leaves_cache:
        leaves = []
        def _exit_term(term: Term, parent: Term):
            if len(term.args) == 0:
                leaves.append(term)
        postorder_traversal(root, lambda term: (), _exit_term)
        leaves_cache[root] = leaves
    leaves = leaves_cache[root]
    if operator is None:
        return leaves
    filtered_leaves = [child_term for child_term in leaves if child_term.operator == operator]
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
    return all(t.operator == ts[0].operator and len(t.args) == len(ts[0].args) for t in ts)

def copy_term(term: Term) -> Term:
    ''' Deep copy of term without bindings '''
    def fn(t: Term, args: list[Term]) -> Term:
        return Term(t.operator, args, {})
    return postorder_map(term, fn)

#NOTE: on same unifications - unification cache? - we do not implement it here, it is done on term tries - see syntax py

def replace_star_wildcard(term: Term, count: int, wildcard: str = "*", to_wildcard: str = "?") -> Term:
    if len(term.args) == 0 or term.args[-1].operator != wildcard:
        return term
    term_copy = copy_term(term)
    while len(term_copy.args) > 0 and term_copy.args[-1].operator == wildcard:
        term_copy.args.pop()
    num_create = count - len(term_copy.args)
    for _ in range(num_create):
        term_copy.args.append(Term(to_wildcard))
    return term_copy    

def unify(b: UnifyBindings, is_equiv: Callable, *terms: Term) -> bool:
    ''' Unification of terms. Uppercase leaves are meta-variables, 
        ? is wildcard leaf - should not be used as operation
        * is wildcard args - 0 or more

        Note: we do not check here that bound meta-variables recursivelly resolve to concrete terms.
        This should be done by the caller.
    '''
    filtered_terms = [t for t in terms if t.operator != "?"]
    if len(filtered_terms) < 2:
        return True
    t_is_meta = [t.operator.isupper() for t in filtered_terms]
    meta_operators = set([t.operator for t, is_meta in zip(filtered_terms, t_is_meta) if is_meta])
    meta_terms = b.get(*meta_operators)
    bound_meta_terms = [bx for bx in meta_terms if bx is not None]
    concrete_terms = [t for t, is_meta in zip(filtered_terms, t_is_meta) if not is_meta]
    all_concrete_terms = bound_meta_terms + concrete_terms
    if len(all_concrete_terms) > 1:
        max_args_count = max(len([a for a in t.args if a.operator != "*"]) for t in all_concrete_terms)
        all_concrete_terms = [replace_star_wildcard(t, max_args_count) for t in all_concrete_terms]
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
    postorder_traversal(term, lambda t: (), _exit_term)
    return eq_terms

def bind_leaves(root: Term, operator: str, bindings: tuple[int, int] | list[tuple[int, int]], leaves_cache = {}):
    ''' Manually asign semantic vector to a specific term in specific evaluation '''
    leaves = get_leaves(root, operator, leaves_cache)
    if len(leaves) == 0:
        return 
    if type(bindings) is tuple:
        bindings = [bindings] * len(leaves)
    for leaf, binding in zip(leaves, bindings):
        if binding is None:
            continue
        eval_id, semantics_id = binding 
        if semantics_id is None:
            leaf.bindings.pop(eval_id, None)
        else:
            leaf.bindings[eval_id] = semantics_id

# t1 = Term("+", [Term("x"), Term("*", [Term("x"), Term("x")])])
# bind_leaves(t1, "x", [None, None, (0, 0)])
# res = term_to_str(t1)
pass

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

def parse_term(term_str: str, i: int = 0) -> tuple[Term, int]:
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
            term = Term(name, args, bindings)
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
            leaf = Term(literal)
            branches[0].append(leaf)
    return branches[0][0], i

def evaluate(eval_id, term: Term, ops: dict[str, Callable], semantic_index: SemanticIndex) -> Any:
    ''' Fully or partially evaluates term (concrete or abstract) '''
    def _eval(term: Term, args: list[Any]):
        if eval_id not in term.bindings:
            if term.operator in ops and all(arg is not None for arg in args):
                semantics = ops[term.operator](*args)
                semantics_id = semantic_index.add(semantics)
                term.bindings[eval_id] = semantics_id
            else:
                return None
        return semantic_index.get(term.bindings[eval_id])
    res_semantics = postorder_map(term, _eval)
    return res_semantics

if __name__ == "__main__":
    # tests
    t1, _ = parse_term("(f (f X (f x (f x)) (f x (f x))))")
    t2, _ = parse_term("(f (f (f x x) Y Y))")
    t3, _ = parse_term("(f Z)")
    b = UnifyBindings()
    res = unify(b, points_are_equiv, t1, t2, t3)
    pass


    t1, _ = parse_term("(f (f (f x x) (f x (f x)) (f x (f x))))")
    # t1, _ = parse_term("(f x y z)")
    # p1, _ = parse_term("(f (f X X) Y Y)")
    p1, _ = parse_term("(f ? ? *)")
    matches = match_term(t1, p1)
    matches = [(term_to_str(m[0]), {k:term_to_str(v) for k, v in m[1].bindings.items()}) for m in matches]
    pass


    # res, _ = parse_term("  \n(   f   (g    x :0:1)  (h \nx) :0:12)  \n", 0)
    res, _ = parse_term("  \n(   f   (g    x)  (h \nx))  \n", 0)
    bind_leaves(res, "g", (0, None))
    res2 = term_to_str(res)
    pass 

    sem_idx = DummySemanticIndex()
    sem_id1 = sem_idx.add(1)
    eval_id = 0
    bind_leaves(res, "x", (eval_id, sem_id1), leaves_cache={})
    print(term_to_str(res))
    ev1 = evaluate(0, res, {"f": lambda x, y: x + y, "g": lambda x: x * 2, "h": lambda x: x ** 2}, sem_idx)
    pass    