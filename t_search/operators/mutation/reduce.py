from typing import TYPE_CHECKING, Callable

import torch
from .base import TermMutation
from syntax import Op, Term, Value, Variable, parse_term
from syntax.traverse import postorder_traversal, TRAVERSAL_EXIT_NODE
import sympy as sp

if TYPE_CHECKING:
    from t_search.solver import GPSolver

sp_alg_ops_f = {
    "add": sp.Add,
    "mul": sp.Mul,
    "pow": sp.Pow,
    "neg": lambda a: -a,
    "inv": lambda a: 1/a,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos
}

sp_alg_ops_b = {
    sp.Add: "add",
    sp.Mul: "mul",
    sp.Pow: "pow"
}

def to_sympy(root: Term, ops: dict[str, Callable] = sp_alg_ops_f) -> sp.Expr:

    args_stacks = [[]]

    def enter_args(term: Term, *_):
        if term.arity() == 0:
            if isinstance(term, Value):
                if torch.is_tensor(term.value):
                    sp_value = sp.Float(term.value.item()) # value is the const tensor
                else:
                    sp_value = sp.Float(term.value) # value is the const tensor 
            elif isinstance(term, Variable):
                sp_value = sp.Symbol(term.var_id)
            args_stacks[-1].append(sp_value)
            return TRAVERSAL_EXIT_NODE
        args_stacks.append([])

    def exit_term(term: Term, *_):
        args = args_stacks.pop()
        assert isinstance(term, Op)
        if term.op_id in ops:
            sp_value = ops[term.op_id](*args)
        else:
            sp_value = sp.Function(term.op_id)(*args)
        args_stacks[-1].append(sp_value)

    postorder_traversal(root, enter_args, exit_term)

    sp_expr = args_stacks[0][0]

    return sp_expr

def from_sympy(root: sp.Expr, op_mapping: dict[sp.Expr, str] = sp_alg_ops_b,
                alloc_val: Callable = lambda v: Value(v),
                alloc_var: Callable = lambda var_id: Variable(var_id),
                alloc_op: Callable = lambda op_id: (lambda *args: Op(op_id, args))
                ) -> Term:
    if root.is_Symbol:
        return alloc_var(str(root))
    if root.is_Number:
        return alloc_val(float(root))
    if root.is_Mul and -1 in root.args:
        args_wo_minus = [a for a in root.args if a != -1]
        if len(args_wo_minus) == 1:  # just -x
            arg = from_sympy(args_wo_minus[0], op_mapping, alloc_val, alloc_var, alloc_op)
            op_res =  alloc_op("neg")(arg)
            return op_res 
    if root.is_Pow and root.exp == -1:
        arg = from_sympy(root.base, op_mapping, alloc_val, alloc_var, alloc_op)
        op_res = alloc_op("inv")(arg)
        return op_res
    
    if root.is_Function and root.func not in op_mapping:
        args = [from_sympy(a, op_mapping, alloc_val, alloc_var, alloc_op) for a in root.args]
        op = alloc_op(str(root.func))(*args)
        return op
    
    # op is assumed 
    args = [from_sympy(a, op_mapping, alloc_val, alloc_var, alloc_op) for a in root.args]
    op = alloc_op(op_mapping[root.func])(*args)
    return op

def sp_simplify(term: Term, *, 
                to_dict: dict = sp_alg_ops_f,
                from_dict: dict = sp_alg_ops_b,
                alloc_val: Callable = lambda v: Value(v),
                alloc_var: Callable = lambda var_id: Variable(var_id),
                alloc_op: Callable = lambda op_id: lambda *args: Op(op_id, args)) -> Term:
    sp_expr = to_sympy(term, to_dict)
    sp_expr_simple = sp.simplify(sp_expr)
    term_simple = from_sympy(sp_expr_simple, from_dict, alloc_val, alloc_var, alloc_op)
    return term_simple    
    
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

if __name__ == "__main__":

    t1, _ = parse_term("(mul (add (neg 7) (mul 0.51 (add x x))) (add 5 4))")
    # print(t1)
    # t1, _ = parse_term("(mul (mul 2 (add x 1)) (inv (add x 1)))")
    # t1, _ = parse_term("(neg (neg (neg x)))")
    # t1_expr = to_sympy(t1)
    print(t1)    
    t2 = sp_simplify(t1)
    print(t2)
    pass