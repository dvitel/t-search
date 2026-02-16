import torch
from t_search.operators.geometric.base import BaseGeometricMutation
from t_search.syntax import Term, Value

class SemanticGeometricMutation(BaseGeometricMutation):
    ''' Implementing Semantic Geometric Mutation from Moraglio 2012 
        Parent program is lineary combined with random term 

        p' = p + r * (t1 - t2)
        r - random const 
        t1, t2 - random terms 
    '''
    def __init__(self, *, 
                    min_grow_depth = 3,
                    max_grow_depth = 5, 
                    **kwargs):
        super().__init__(**kwargs)
        self.min_grow_depth = min_grow_depth
        self.max_grow_depth = max_grow_depth        

    def mutate_term(self, term: Term) -> Term | None:

        final_term = None

        # if term not in self.terms:
        #     f = self.semantics.get_outputs(term)
        #     ind = Ind(term=term, outputs=f.clone(), weights=torch.zeros((self.num_funcs), 
        #                                                                 dtype=self.target.dtype, 
        #                                                                 device=self.target.device), 
        #                                                                 funcs=[])
        #     self.terms[term] = ind
        # ind = self.terms[term]
        
        # d1 = self.rnd.integers(self.min_grow_depth, self.max_grow_depth+1)
        for _ in range(10):
            t1 = self.syntax.grow(self.max_grow_depth)
            if self.syntax.get_depth(t1) >= self.min_grow_depth:
                break 
        # d2 = self.rnd.integers(self.min_grow_depth, self.max_grow_depth+1)
        for _ in range(10):
            t2 = self.syntax.grow(self.max_grow_depth)
            if self.syntax.get_depth(t2) >= self.min_grow_depth:
                break 

        # funcs_to_place = []
        # func_ids = []
        # for tt in [t1, t2]:
        #     idx = next((i for i, f in enumerate(ind.funcs) if f == tt), None)
        #     if idx is None:
        #         if len(ind.funcs) < self.max_funcs:
        #             func_ids.append(len(ind.funcs))
        #             ind.funcs.append(tt)
        #         else:
        #             funcs_to_place.append(tt)
        #     else:
        #         func_ids.append(idx)

        # if len(funcs_to_place) > 0:
        #     funcs_to_remove = []
        #     removed_weights = None
        #     removed_weights = torch.zeros(len(funcs_to_place), dtype=self.target.dtype, device=self.target.device)
        #     sort_ids = torch.argsort(ind.weights) # remove least contributing funcs
        #     for i, tt in enumerate(funcs_to_place):
        #         new_id = sort_ids[i].item()
        #         func_to_remove = ind.funcs[new_id]
        #         func_ids.append(new_id) # where to place new terms
        #         ind.funcs[new_id] = tt # replace with new term
        #         funcs_to_remove.append(func_to_remove)
        #         removed_weights[i] = ind.weights[new_id].item() # save removed weight for later
        #         ind.weights[new_id] = 0.0 # reset weight for new term
        #     removed_vectors = self.semantics.get_outputs(funcs_to_remove, return_type="tensor")
        #     removed_sum_vector = (removed_weights.unsqueeze(-1) * removed_vectors).sum(dim=0)
        #     ind.outputs -= removed_sum_vector # remove contribution of removed funcs from outputs
        #     del removed_vectors        

        # e1 = self.evaluator.eval(trimmed_term) # get vector for t1
        # e2 = self.evaluator.eval(term) # get vector for t2
        # assert torch.allclose(e1[1], e2[1], atol=1e-2, rtol=1e-2)

        self.evaluator.eval([term, t1, t2]) # get vectors 
        if not self.semantics.is_valid(t1) or not self.semantics.is_valid(t2):
            return None
        
        if self.use_best_epsilon:
            s1, s2 = self.semantics.get_outputs([t1, t2], return_type="list")

            sd = s1 - s2 
            sd2 = (sd * sd).sum()

            f = self.semantics.get_outputs(term)
        

            best_epsilon = ((self.target - f) * sd).sum() / (sd2 + 1e-8)
            best_epsilon.clamp_(-self.epsilon, self.epsilon)

        else:

            best_epsilon = self.rnd.random() * self.epsilon

        # assert torch.isfinite(best_epsilon)

        # ind.weights[func_ids[0]] += best_epsilon
        # ind.weights[func_ids[1]] -= best_epsilon
        # ind.outputs += best_epsilon * sd

        # NOTE: we still build the term to make comparison of methods fair
        # f = self.semantics.get_outputs(term)

        t1_term = self.syntax.get_op("mul", self.syntax.get_const(value = best_epsilon), t1)
        t2_term = self.syntax.get_op("mul", self.syntax.get_const(value = -best_epsilon), t2)

        final_term = self.syntax.get_op("add", term, self.syntax.get_op("add", t1_term, t2_term))

        trimmed_term = self.trim_deep_term(final_term)
        if isinstance(trimmed_term, Value):
            return None

        self.evaluator.eval(trimmed_term)
        
        # final_term = self.trim_deep_term(mutated_term)
        # We do not validate here the depth of the tree
        # if self.check_validity and not self.syntax.is_valid(final_term):
        #     return None
        return trimmed_term 