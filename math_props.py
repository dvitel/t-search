''' Math properties of terms. 
    Main goal is to cut search space (SS) considering the properties. 
    Empirically, we want to check how tracking property improves performance. 

    Major routine groups:
    0. Property representation and storage. 
    1. Establishing property given evaluation history. 
    2. Reduction of a term based on property. 
    
    Props:
    1. TODO Idempotence: f(f(*)) = f(*)
        SS cut: f is not allowed under f 
    2. TODO Commutativity: f(x, y) = f(y, x)
        SS cut: orders (x, y): y is bigger or equal to x.
                if y smaller during synth - swap x and y
    3. TODO Monotonicity: f(x) <= f(y) if x <= y
        SS cut: defines direction of change towards target: 
                f(x) < t ==> pick y > x to approach t
    4. TODO Constancy: f(x) = c, forall x 
        SS cut: x represents the intron, useless part of term, to need to visit it.
    --
    TODO Add more with justification of usefulness for search performance

    ??? As we work in multidim space (1 test is one dim), properties are established on per dimension basis ??? 
    No, the props are props of symbol and its arguments. 
'''

from dataclasses import dataclass
from typing import Optional

@dataclass
class Idempotence:


@dataclass 
class TermProps:
    idempotent: Optional[bool] = None
    commutative: Optional[bool] = False
    monotonic: Optional[bool] = False

def is_idempotent(term, position, semantics):
    pass 

def reduce_idempotent(term):
    pass 