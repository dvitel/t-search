
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


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
