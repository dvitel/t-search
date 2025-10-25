from typing import TYPE_CHECKING
from .base import TermMutation
from syntax import sp_alg_ops_f, sp_alg_ops_b, sp_simplify
from term import Term

if TYPE_CHECKING:
    from gp import GPSolver
    
class Reduce(TermMutation): 
    ''' Syntactic Simplifier based on domain axioms '''

    def  __init__(self, name: str = "syn_simpl", *,
                    to_ops: dict = sp_alg_ops_f, from_ops: dict = sp_alg_ops_b,
                    check_validity: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.to_ops = to_ops
        self.from_ops = from_ops
        self.check_validity = check_validity
    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        new_term = sp_simplify(term, to_dict = self.to_ops, from_dict=self.from_ops,
                               alloc_val = lambda value: solver.const_builder.fn(value = value),
                               alloc_var = lambda var_id: solver.var_builder.fn(var_id = var_id),
                               alloc_op = lambda op_id: lambda *args: solver.op_builders.get(op_id).fn(*args))
        if self.check_validity and not solver.is_valid(new_term):
            return None
        return new_term
