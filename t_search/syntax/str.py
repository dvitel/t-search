''' String utils for Terms: parse and pretty-printing '''

from collections import deque
from typing import Optional, Sequence

import torch

from .traverse import postorder_map

from .term import LeafStructure, NonLeafStructure, Term, Value, Variable, Op, MetaVariable, OpWildcard, Wildcard, Wildcards


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

