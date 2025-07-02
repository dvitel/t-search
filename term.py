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

import torch

@dataclass(eq=False, unsafe_hash=False)
class Term:
    def get_args(self) -> list['Term']:
        return []
    def get_bindings(self) -> dict[int, int]:
        return {}

@dataclass(eq=False, unsafe_hash=False)
class Leaf(Term):
    name: str
    bindings: dict[int, int] = field(default_factory=dict)
    ''' Binding defines index in semantics, the vector which specifies evaluation of the term 
        Map: evaluation id to semantics id 
    '''
    def get_bindings(self):
        return self.bindings
    
@dataclass(eq=False, unsafe_hash=False)
class Branch(Term):
    operator: str
    args: list[Term] = field(default_factory=list)
    def get_args(self):
        return self.args


@dataclass(eq=False, unsafe_hash=False)
class BoundBranch(Branch):
    bindings: dict[int, int] = field(default_factory=dict)
    ''' Binding defines index in semantics, the vector which specifies evaluation of the term '''
    def get_bindings(self):
        return self.bindings
    
def postorder_traversal(term: Term, depth: int = 0) -> Generator[Term, Any, None]:
    processed_args = []
    for arg in term.get_args():
        processed_arg = yield from postorder_traversal(arg, depth + 1)
        if processed_arg is not None:
            processed_args.append(processed_arg)
    yield (term, processed_args, depth)

# def postorder_map_lst(term: Term, procs: list[tuple[Type[Term], Callable]]) -> Any:
    # fs = [prc for tp, prc in procs if isinstance(term, tp)]
    # if len(fs) == 0:
    #     return term 
    # args = [postorder_map_lst(arg, procs) for arg in term.get_args()]
    # new_term = fs[0](term, args)
    # return new_term

def postorder_map(term: Term, procs: dict[Type[Term], Callable]) -> Any:    
    scored_procs = [(tp, prc, sum(1 for tp1 in procs.keys() if issubclass(tp, tp1))) for tp, prc in procs.items()]
    scored_procs.sort(key=lambda x: x[-1], reverse=True)
    procs_lst = [(tp, prc) for tp, prc, _ in scored_procs]   
    prev_term = None     
    terms = postorder_traversal(term)
    while True: 
        try:
            term, args, depth = terms.send(prev_term)
            fs = [prc for tp, prc in procs_lst if isinstance(term, tp)]
            if len(fs) == 0:
                prev_term = term
            else:
                prev_term = fs[0](term, args, depth)
        except StopIteration as e:
            return prev_term        

# def val_to_str(v: Any) -> str:
#     if isinstance(v, Term):
#         term_to_str(v)
#     if type(v) is str:
#         return ("'" + v + "'") 
#     return str(v)

def leaf_to_str(term: Leaf, _):
    binds = [f"{eid}:{sid}" for eid, sid in term.get_bindings().items()]
    if len(binds) == 0:
        return term.name
    binds = [term.name, *binds]
    return "[" + " ".join(binds) + "]"

def branch_to_str(term: Branch, args: list[str]):
    return "(" + " ".join([term.operator, *args]) + ")"

def bound_branch_to_str(term: BoundBranch, args: list[str]):
    binds = [f"{eid}:{sid}" for eid, sid in term.get_bindings().items()]
    binds = [term.operator, *binds]
    operator = "[" + " ".join(binds) + "]"
    return "(" + " ".join([operator, *args]) + ")"

# NOTE: idea in this str method was to be concise also - for debugging usage, not fully human readable though
#       yet we can restore (not all possible terms) many terms from this string form with parse
def term_to_str(term: Term) -> str: 
    ''' LISP style string '''
    res = postorder_map(term, {Branch: branch_to_str, Leaf: leaf_to_str, BoundBranch: bound_branch_to_str})
    return res 

def skip_spaces(term_str: str, i: int) -> int:
    while i < len(term_str) and term_str[i].isspace():
        i += 1
    return i

def skip_till_break(term_str: str, j: int, breaks) -> int:
    while j < len(term_str) and term_str[j] not in breaks:
        j += 1    
    return j

def parse_binding_or_literal(term_str: str, i: int = 0): 
    i = skip_spaces(term_str, i)
    assert i < len(term_str), f"Expected binding or literal at position {i} in term string: {term_str}"
    if term_str[i] == '[':
        j = skip_till_break(term_str, i + 1, "]")
        assert j < len(term_str) and term_str[j] == ']', f"Expected ']' at position {j} in term string: {term_str}"
        binding_str = term_str[i + 1:j].strip().split(" ")
        name = binding_str[0]
        values = {int(vs[0]):int(vs[1]) for s in binding_str[1:] if s for vs in [s.strip().split(":")]}        
        return (name, values), j + 1
    else: #literal
        j = skip_till_break(term_str, i + 1, " )")
        literal = term_str[i:j]
        assert literal, f"Literal cannot be empty at position {i}:{j} in term string: {term_str}"
        return (literal, {}), j

def parse_term(term_str: str, i: int = 0) -> tuple[Term, int]:
    ''' Read term from string, return term and end of term after i '''
    branches = deque([[]])
    while True:
        i = skip_spaces(term_str, i)
        if i >= len(term_str):
            break
        if term_str[i] == ')': # end of branch - stop reading args 
            cur_term = branches.popleft() # should contain bindings and args 
            name, bindings = cur_term[0]
            args = cur_term[1:]
            if len(bindings) > 0:
                term = BoundBranch(name, args, bindings)
            else:
                term = Branch(name, args)
            branches[0].append(term)
            i += 1            
        elif term_str[i] == '(': # branch
            binding, i = parse_binding_or_literal(term_str, i + 1)
            branches.appendleft([binding])
        else: #leaf
            (name, values), i = parse_binding_or_literal(term_str, i)
            # terms.appendleft([binding])
            leaf = Leaf(name, values)
            branches[0].append(leaf)
    return branches[0][0], i

# res, _ = parse_term("  \n(   f   (g    x)  (h \nx))  \n", 0)
# pass 

def get_leaves(root: Term, name: Optional[str] = None, leaves_cache = {}) -> list[Leaf]:
    ''' Find all leafs in root that are equal to term by name (ignoring bindings) '''
    if root not in leaves_cache:
        leaves = []
        for child_term, _, _ in postorder_traversal(root):
            if isinstance(child_term, Leaf):
                leaves.append(child_term)
        leaves_cache[root] = leaves
    leaves = leaves_cache[root]
    if name is None:
        return leaves
    filtered_leaves = [child_term for child_term in leaves if child_term.name == name]
    return filtered_leaves

def bind_leaves(root: Term, name: str, eval_id: int, semantics_id: int, leaves_cache = {}):
    ''' Manually asign semantic vector to a specific term in specific evaluation '''
    for leaf in get_leaves(root, name, leaves_cache):
        leaf.bindings[eval_id] = semantics_id

alg_torch = {
    "+": lambda a, b: a + b,
    "*": lambda a, b: a * b,
    "**": lambda a, b: a ** b,
    "0-": lambda a: -a,
    "1/": lambda a: 1 / a,
    "exp": lambda a: torch.exp(a),
    "log": lambda a: torch.log(a),
    "sin": lambda a: torch.sin(a),
    "cos": lambda a: torch.cos(a),
}

def evaluate(term: Term, ops: dict[str, Callable], leaves: dict[str, Callable]) -> torch.Tensor:
    ''' Ffully or partially evaluates term (concrete or abstract) '''
    def eval_op(args, op, isOp):
        if isOp:
            if op in ops and all(type(a) is EvalTerm for a in args):
                value = ops[op](*(a.value for a in args))
                return EvalTerm( )
            else:
                return 
        else:

        if op == "x":
            return free_vars[args[0]]
        elif op == "c":
            return args[0]
        elif op == "+":
            return args[0] + args[1]
        elif op == "*":
            return args[0] * args[1]
        elif op == "**":
            return args[0] ** args[1]
        elif op == "0-":
            return -args[0]
        elif op == "1/":
            return 1 / args[0]
        elif op == "exp":
            return np.exp(args[0])
        elif op == "log":
            return np.log(args[0])
        elif op == "sin":
            return np.sin(args[0])
        elif op == "cos":
            return np.cos(args[0])
        else:
            return op
    res = postorder_traversal(term_instance, eval_op)
    return res
