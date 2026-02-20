''' Basic for geometric operators '''

from collections import deque
from dataclasses import dataclass, field

import torch
from t_search.base import ServiceBase
from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.semantics import Semantics
from t_search.operators.mutation import TermMutation
from t_search.operators.reduction import LincombMixin
from t_search.syntax.term import Op, Term, Value
    
class BaseGeometricMutation(TermMutation, ServiceBase, LincombMixin): 

    def __init__(self, *, 
                    fitness: Fitness,
                    semantics: Semantics,
                    evaluator: Evaluator,
                    target: torch.Tensor,
                    epsilon = 0.02, 
                    alpha: float = 0.3,
                    identity_atol: float = 1e-2,
                    identity_rtol: float = 1e-2,
                    with_reduction: bool = True, 
                    use_best_epsilon: bool = True,
                    torch_gen: torch.Generator | None = None,
                    **kwargs):
        super().__init__(**kwargs)
        self.fitess = fitness
        self.semantics = semantics
        self.epsilon = epsilon
        self.target = target
        self.evaluator = evaluator
        self.alpha = alpha
        self.identity_atol = identity_atol
        self.identity_rtol = identity_rtol
        self.with_reduction = with_reduction    
        self.use_best_epsilon = use_best_epsilon
        self.torch_gen = torch_gen

    def trim_term(self, term: Term) -> Term:
        term_depth = self.syntax.get_depth(term)
        if  term_depth >= self.syntax.max_term_depth: 
            max_pos_depth = term_depth // 2 
            pos_terms = []
            for pos in self.syntax.get_positions(term):
                if self.syntax.get_depth(pos.term) <= max_pos_depth and not isinstance(pos.term, Value):
                    pos_terms.append(pos.term)
            pos_fitness = self.fitess.get_fitness(pos_terms, return_type="tensor")
            best_pos_id = torch.argmin(pos_fitness).item()            
            del pos_fitness
            term = pos_terms[best_pos_id]
        return term 

    def trim_deep_term(self, term: Term, multiplier: torch.Tensor | None = None, should_trim: bool = True) -> Term: 
        # linearize all adds at root and remove smallest coefficients       
        if not (isinstance(term, Op) and term.op_id == "add"):
            return term
        multiplier = multiplier if multiplier is not None else self.syntax.one_value.value
        new_args = []
        groups: dict[Term, float] = {} 
        q = deque([term])
        while len(q) > 0:
            t = q.popleft()
            if isinstance(t, Op) and t.op_id == "add":
                for arg in t.get_args():
                    q.append(arg)
                continue
            elif isinstance(t, Op) and t.op_id == "mul":
                t_args = t.get_args()
                const_id = next((i for i, arg in enumerate(t_args) if isinstance(arg, Value)), None)
                if const_id is not None:
                    const_value = t_args[const_id].value 
                    other_arg = t_args[1 - const_id]
                    if isinstance(other_arg, Op) and other_arg.op_id == "add":
                        trimmed_other_arg = self.trim_deep_term(other_arg, multiplier=const_value, should_trim = False)
                        q.append(trimmed_other_arg)
                    else:
                        groups[other_arg] = groups.get(other_arg, 0.0) + const_value
                    continue
                else:
                    groups[t] = groups.get(t, 0.0) + self.syntax.one_value.value   
                    continue     
            groups[t] = groups.get(t, 0.0) + self.syntax.one_value.value
        for k, w in groups.items():
            assert not(isinstance(k, Op) and k.op_id == "add")
            w *= multiplier
        # num_to_remove = (self.syntax.get_depth(term) - self.syntax.max_term_depth) // 2 # (1 add and 1 mul)
        if should_trim:
            while len(groups) > 1:
                max_depth = max(self.syntax.get_depth(t) for t in groups.keys()) + 2
                if max_depth < self.syntax.max_term_depth:
                    break
                min_weight = float('inf')
                min_term = None
                for t, w in groups.items():
                    if abs(w) < min_weight:
                        min_weight = abs(w)
                        min_term = t
                assert min_term is not None
                groups.pop(min_term, None)    
        for t, w in groups.items():
            assert torch.isfinite(w)
            if self.mul_id_fn(w): # mul 1
                new_args.append(t)
            elif self.add_id_fn(w): # mull 0 
                continue
            elif isinstance(t, Value):
                new_args.append(self.syntax.get_const(value=w * t.value))
            else:
                new_args.append(self.syntax.get_op("mul", self.syntax.get_const(value=w), t))
        if len(new_args) == 0:
            return self.syntax.zero_value
        if len(new_args) == 1:
            return new_args[0]
        new_const_value = self.syntax.zero_value.value.clone()
        non_consts = []
        for arg in new_args:
            if isinstance(arg, Value):
                new_const_value += arg.value
            else:
                non_consts.append(arg)
        if not self.add_id_fn(new_const_value): 
            non_consts.append(self.syntax.get_const(value=new_const_value))
        return self.syntax.get_op("add", *non_consts)

