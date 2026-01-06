from typing import Callable

import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.operators.mutation import TermMutation
from t_search.syntax.syntax import Syntax
from t_search.syntax import Op, Term, Value, Variable, parse_term
from t_search.syntax.traverse import postorder_traversal, TRAVERSAL_EXIT_NODE
import sympy as sp

from t_search.utils import timed

sp_alg_ops_f = {
    "add": sp.Add,
    "sub": lambda a, b: a - b,
    "mul": sp.Mul,
    "div": lambda a, b: a / b,
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
                alloc_op: Callable = lambda op_id: (lambda *args: Op(op_id, args)),
                use_unary: bool = True,
                use_pow: bool = True) -> Term | None:
    if root.is_Symbol or root.is_symbol:
        return alloc_var(str(root))
    if root.is_Number or root.is_number:
        if not root.is_real:
            return None
        return alloc_val(float(root))
    if root.is_Mul and -1 in root.args:
        args_wo_minus = [a for a in root.args if a != -1]
        if len(args_wo_minus) == 1:  # just -x
            arg = from_sympy(args_wo_minus[0], op_mapping, alloc_val, alloc_var, alloc_op, use_unary=use_unary, use_pow=use_pow)
            if arg is None:
                return None
            if use_unary:
                op_res = alloc_op("neg")(arg)
            else: # asssumes sub in func set 
                op_res = alloc_op("sub")(alloc_val(0), arg)
            return op_res 
    if root.is_Pow and root.exp == -1:
        arg = from_sympy(root.base, op_mapping, alloc_val, alloc_var, alloc_op, use_unary=use_unary, use_pow=use_pow)
        if arg is None:
            return None
        if use_unary:
            op_res = alloc_op("inv")(arg)
        else: # assumes div in func set
            op_res = alloc_op("div")(alloc_val(1), arg)
        return op_res
    
    if root.is_Function and root.func not in op_mapping:
        args = []
        for a in root.args:
            arg = from_sympy(a, op_mapping, alloc_val, alloc_var, alloc_op, use_unary=use_unary, use_pow=use_pow) 
            if arg is None:
                return None
            args.append(arg)
        func_name = str(root.func)
        if func_name == "tan": # go back to sin/cos
            if use_unary:
                op = alloc_op("mul")(alloc_op('sin')(*args), alloc_op("inv")(alloc_op("cos")(*args)))
            else:
                op = alloc_op("div")(alloc_op('sin')(*args), alloc_op("cos")(*args))
        else:
            op = alloc_op(func_name)(*args)
        # assert len(op.args) <= 2
        return op
    
    # op is assumed 
    args = []
    for a in root.args:
        arg = from_sympy(a, op_mapping, alloc_val, alloc_var, alloc_op, use_unary=use_unary, use_pow=use_pow)
        if arg is None:
            return None
        args.append(arg)
    if not use_pow and (op_mapping[root.func] == "pow"): # express pow through exp log
        # pow(x, y) = exp(y * log(x))
        arg1 = alloc_op("log")(args[0])
        arg2 = args[1]
        mul_op = alloc_op("mul")(arg2, arg1)
        op = alloc_op("exp")(mul_op)
    else:
        op = alloc_op(op_mapping[root.func])(*args)
    # assert len(op.args) <= 2
    return op

def sp_simplify(term: Term, *, 
                to_dict: dict = sp_alg_ops_f,
                from_dict: dict = sp_alg_ops_b,
                alloc_val: Callable = lambda v: Value(v),
                alloc_var: Callable = lambda var_id: Variable(var_id),
                alloc_op: Callable = lambda op_id: lambda *args: Op(op_id, args),
                use_unary:bool = True,
                use_pow:bool = True) -> Term | None:
    sp_expr = to_sympy(term, to_dict)
    sp_expr_simple = sp.simplify(sp_expr)
    term_simple = from_sympy(sp_expr_simple, from_dict, alloc_val, alloc_var, alloc_op, use_unary=use_unary, use_pow=use_pow)
    return term_simple
    
class Reduce(TermMutation): 
    ''' Syntactic Simplifier based on domain axioms '''

    def  __init__(self, *,
                    evaluator: Evaluator,
                    to_ops: dict = sp_alg_ops_f, from_ops: dict = sp_alg_ops_b,
                    check_validity: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.to_ops = to_ops
        self.from_ops = from_ops
        self.check_validity = check_validity
        self.evaluator = evaluator
    def mutate_term(self, term: Term) -> Term | None:
        def simplify():
            new_term = sp_simplify(term, to_dict = self.to_ops, from_dict=self.from_ops,
                                alloc_val = lambda value: self.syntax.get_const(value = value),
                                alloc_var = lambda var_id: self.syntax.get_var(var_id = var_id),
                                alloc_op = lambda op_id: lambda *args: self.syntax.get_op(op_id, *args),
                                use_unary=self.syntax.has_op("inv"),
                                use_pow=self.syntax.has_op("pow"))
            return new_term
        new_term, elapsed = timed(simplify)()
        self.add_metrics(simplify_time = elapsed)
        if new_term is None:
            return None
        if self.check_validity:
            is_valid, v_elapsed = timed(self.syntax.is_valid)(new_term)
            self.add_metrics(validity_check_time = v_elapsed)
            if not is_valid:
                return None
        # if self.debug:
        #     eval_before = self.evaluator.eval(term)
        #     eval_after = self.evaluator.eval(new_term)
        #     existing_eval_before = eval_before[1]
        #     existing_eval_after = eval_after[1]
        #     before_mask = torch.isfinite(existing_eval_before)
        #     after_mask = torch.isfinite(existing_eval_after)
        #     common_mask = before_mask & after_mask
        #     assert torch.allclose(existing_eval_before[common_mask], existing_eval_after[common_mask], rtol=1, atol=1e-1), f"Simplification changed semantics from {eval_before} to {eval_after}"
        #     pass
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