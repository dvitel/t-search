''' Mixins to work with syntax based on some axioms '''

from collections import deque
from math import prod
from typing import Callable
import torch

from t_search.syntax.term import Op, Term, TermPos, Value
from t_search.utils import optimize_kb, ransac_all_pairs

class LincombMixin: 
    ''' Provides methods to create and reduce linear combinations k * X + b, k, b Values and X is term'''

    def add_id_fn(self, v):
        res = torch.isclose(v, self.syntax.zero_value.value, atol=self.identity_atol, rtol=self.identity_rtol)
        return res
    
    def mul_id_fn(self, v):
        res = torch.isclose(v, self.syntax.one_value.value, atol=self.identity_atol, rtol=self.identity_rtol)
        return res
    
    def is_in_lincomb(self, pos: TermPos) -> bool:
        if pos.parent is None:
            return False
        term = pos.parent.term
        if isinstance(term, Op) and term.op_id in ["add", "mul"]:
            other_arg = term.get_args()[1 - pos.pos]
            if isinstance(other_arg, Value):
                return True 
        return False

    def decompose_lincomb(self, term: Term) -> tuple[Value | None, Term, Value | None]:
        ''' Returns (k, X, b) if term is of the form k * X + b or None otherwise. k, b can be None if they are not present. '''
        if isinstance(term, Op) and term.op_id == "add":
            args = term.get_args()
            if len(args) != 2:
                return None, term, None
            value_id = next((i for i, a in enumerate(args) if isinstance(a, Value)), None)
            if value_id is None:
                return None, term, None
            other_id = 1 - value_id
            b_value = args[value_id]
            other = args[other_id]
            if isinstance(other, Op) and other.op_id == "mul":
                mul_args = other.get_args()
                if len(mul_args) != 2:
                    return (None, other, b_value)
                k_id = next((i for i, a in enumerate(mul_args) if isinstance(a, Value)), None)
                if k_id is None:
                    return (None, other, b_value)
                X_id = 1 - k_id
                k_value = mul_args[k_id]
                X = mul_args[X_id]
                return (k_value, X, b_value)
            return (None, other, b_value)
        elif isinstance(term, Op) and term.op_id == "mul":
            args = term.get_args()
            if len(args) != 2:
                return None, term, None
            value_id = next((i for i, a in enumerate(args) if isinstance(a, Value)), None)
            if value_id is None:
                return None, term, None
            other_id = 1 - value_id
            k_value = args[value_id]
            X = args[other_id]
            return (k_value, X, None)
        return None, term, None

    def reduce_lincomb(self, term: Term, 
                        ops: dict[str, Callable] = {"add": lambda vs: sum(v for v in vs), "mul": lambda vs: prod(v for v in vs)},
                        identities: dict[str, Callable] = {}
                      ) -> Term:
        ''' add/mul for binary is tansformed to one of varying arity and then all constants are combined
            then, we return to binary ops. Top-down transformation.

            Additional rules are applied only when self.with_reduction.
            mul rule: (k1 * X + b1)*(k2 * X + b2) --> k3 * X^2 + k4 * X + b3 (4 to 3 consts)
            add rule: k1 * X + k2 * X --> k3 * X (2 to 1 consts, if X is the same)

            PROBLEMS: reduction leads to the following:
                  1. Removal of potential pathes to the target through new nodes 
                  2. Identity removal to some precision could fluctuate the loss especially when it is small enough  
                  3. A cicle in the lineage could appear 
            Pros: we target to reduce number of constants to optimize staying neutral (relativelly)
        '''
        if not isinstance(term, Op):
            return term
        if term.op_id not in ops:
            reduced_args = []
            for a in term.get_args():
                reduced_a = self.reduce_lincomb(a, ops=ops, identities=identities)
                reduced_args.append(reduced_a)
            new_term = self.syntax.get_op(term.op_id, *reduced_args)
            return new_term
        all_terms = deque([term])
        final_args = []
        while len(all_terms) > 0:
            current = all_terms.popleft()
            if isinstance(current, Op) and (current.op_id == term.op_id):
                for a in current.get_args():                    
                    all_terms.append(a)
            else:
                reduced_current = self.reduce_lincomb(current, ops=ops, identities=identities)
                if isinstance(reduced_current, Op) and (reduced_current.op_id == term.op_id):
                    for a in reduced_current.get_args():                    
                        all_terms.append(a) 
                else:               
                    final_args.append(reduced_current)
        const_terms, non_const_terms = [], []
        for a in final_args:
            (const_terms if isinstance(a, Value) else non_const_terms).append(a)
        # if len(const_terms) == 0: # nothing to reduce - leave as it was 
        #     return term
        # final_const = const_terms[0]
        final_const = None 
        if len(const_terms) == 1:
            final_const = const_terms[0]
        elif len(const_terms) > 1:
            reduce_fn = ops[term.op_id]
            new_const = reduce_fn([c.value for c in const_terms])
            final_const = new_const if isinstance(new_const, Value) else self.syntax.get_const(value=new_const)
        # if (final_const is not None) and ((len(non_const_terms) == 0) or (term.op_id not in identities) or (not identities[term.op_id](final_const.value))):
        add_identity_fn = identities.get("add", lambda _: False)
        mul_identity_fn = identities.get("mul", lambda _: False)
        if term.op_id == "add": # insert const after (mul ...) when possible

            if self.with_reduction:
                # rule 1: grouping ki * X in sum by X 
                decomposed_non_const_terms = [self.decompose_lincomb(t) for t in non_const_terms]
                groups = {}
                for (k,X,b) in decomposed_non_const_terms:
                    assert b is None
                    groups.setdefault(X, []).append((k or self.syntax.one_value).value)

                if len(groups) < len(decomposed_non_const_terms): # some grouped 
                    new_non_const_terms = []
                    for X, ks in groups.items():
                        k = sum(ks)
                        k_value = self.syntax.get_const(value=k)
                        if add_identity_fn(k_value.value): #mul 0
                            continue
                        if mul_identity_fn(k_value.value): #mul 1
                            new_X = X
                        else:
                            new_X = self.syntax.get_op("mul", k_value, X)
                        new_non_const_terms.append(new_X)
                    non_const_terms = new_non_const_terms

            if final_const is not None and not add_identity_fn(final_const.value): #add 0
                if mul_identity_fn(final_const.value): #add 1
                    final_const = self.syntax.one_value
                found_id = next((i for i, t in enumerate(non_const_terms) 
                                    if isinstance(t, Op) and t.op_id == "mul" and \
                                    any(isinstance(marg, Value) for marg in t.get_args())), None)
                if found_id is None:
                    non_const_terms.append(final_const)        
                else:    
                    # non_const_terms.insert(found_id + 1, final_const)
                    subterm = self.syntax.get_op("add", non_const_terms[found_id], final_const)
                    non_const_terms = [subterm, *non_const_terms[:found_id], *non_const_terms[found_id + 1:]]
            if len(non_const_terms) == 0:
                non_const_terms.append(self.syntax.zero_value)
        elif term.op_id == "mul": 
            if final_const is not None and not mul_identity_fn(final_const.value): #mul 1
                if add_identity_fn(final_const.value): #mul 0
                    non_const_terms = [self.syntax.zero_value]
                elif self.with_reduction:
                    # rule 1: try to find (k * X + b) in args to add the constant there
                    decomposed_non_const_terms = [self.decompose_lincomb(t) for t in non_const_terms]                
                    found_id, k, X, b = next(((i, k, X, b) for i, (k, X, b) in enumerate(decomposed_non_const_terms) 
                                                if k is not None and b is not None), (None, None, None, None))
                    if found_id is not None:
                        new_k = self.syntax.get_const(value=k.value * final_const.value)
                        new_b = self.syntax.get_const(value=b.value * final_const.value)
                        if add_identity_fn(new_k.value): #mul 0
                            new_term = new_b
                        else:
                            if add_identity_fn(new_b.value): #add 0
                                new_b = None
                            if mul_identity_fn(new_k.value): #mul 1
                                new_k = None
                            if new_k is None and new_b is None:
                                new_term = X
                            elif new_k is None:
                                new_term = self.syntax.get_op("add", X, new_b)
                            elif new_b is None:
                                new_term = self.syntax.get_op("mul", new_k, X)
                            else:
                                new_term = self.syntax.get_op("add", self.syntax.get_op("mul", new_k, X), new_b)
                        non_const_terms[found_id] = new_term
                    else:
                        found_id, X, b = next(((i, X, b) for i, (k, X, b) in enumerate(decomposed_non_const_terms) 
                                                if k is None and b is not None), (None, None, None))
                        if found_id is not None:
                            new_k = final_const
                            new_b = self.syntax.get_const(value=b.value * final_const.value)
                            new_term = self.syntax.get_op("add", self.syntax.get_op("mul", new_k, X), new_b)
                            non_const_terms[found_id] = self.reduce_lincomb(new_term, ops=ops, identities=identities)
                        else:
                            non_const_terms.append(final_const)
                else:
                    non_const_terms.append(final_const)
            
            if len(non_const_terms) == 0:
                non_const_terms.append(self.syntax.one_value)                
        else:
            non_const_terms.append(final_const)      

        if len(non_const_terms) == 1:
            return non_const_terms[0]
        non_const_terms.sort(key=lambda t: self.syntax._get_term_priority(t), reverse=True)
        new_term = self.syntax.get_op(term.op_id, *non_const_terms)
        return new_term

    def build_lincomb(self, k: float, X: Term, b: float) -> Term:
        k_value = self.syntax.get_const(value=k)
        b_value = self.syntax.get_const(value=b)
        if self.add_id_fn(b_value.value): # add 0
            if self.mul_id_fn(k_value.value): # mul 1
                return X
            elif self.add_id_fn(k_value.value): # mul 0
                return self.syntax.zero_value
            else:
                return self.syntax.get_op("mul", k_value, X)
        else:
            if self.mul_id_fn(b_value.value): # add 0
                b_value = self.syntax.one_value
            if self.mul_id_fn(k_value.value): # mul 1
                return self.syntax.get_op("add", X, b_value)
            elif self.add_id_fn(k_value.value): # mul 0
                return b_value
            new_term = self.syntax.get_op("add", self.syntax.get_op("mul", k_value, X), b_value)    
            return new_term    
                    
    def optimize_lincomb(self, candidates: list[Term], targets: torch.Tensor, iters:int=128, sample_size:int=2) -> list[tuple[float, float, Term, float]]:
        X = self.semantics.get_outputs(candidates, return_type="tensor")
        K, B, fit_counts, fit_loss = ransac_all_pairs(X, targets, 
                                                      iters=iters, 
                                                      threshold=0.01,
                                                      min_inliers=10, 
                                                      sample_size=sample_size,
                                                      torch_gen=self.torch_gen) #optimize_kb(X, targets)
        # X_ = K.unsqueeze(-1) * X.unsqueeze(1) + B.unsqueeze(-1) # (n, k, dims) outputs 
        # del X
        # loss_per_term_per_test = self.fitness.get_loss(X_, custom_target=targets) # (n, k, dims) losses 
        # del X_
        # loss_per_term = loss_per_term_per_test.mean(dim=-1) # (n, k) losses
        # del loss_per_term_per_test
        best_fit_id = fit_counts.argmax(dim=1) # (n,) min loss per term
        new_terms = []
        for i, term in enumerate(candidates):
            j = best_fit_id[i].item()
            term_fit_count = fit_counts[i,j].item()
            fl = fit_loss[i,j].item()           
            new_terms.append(((-term_fit_count, fl), K[i,j].item(), term, B[i,j].item()))

        res =  sorted(new_terms, key=lambda x: x[0])
        return res
    
    def optimize_lincomb_batched(self, candidates: list[Term], targets: torch.Tensor, iters:int=128, sample_size:int=2,
                                    candidates_batch: int = 128,
                                    targets_batch: int = 64) -> list[tuple[float, float, Term, float]]:
        new_terms = []
        for i in range(0, len(candidates), candidates_batch):
            cur_candidates = candidates[i:i+candidates_batch]
            X = self.semantics.get_outputs(cur_candidates, return_type="tensor")
            all_Ks, all_Bs, all_fit_counts, all_fit_loss = [], [], [], []
            for j in range(0, targets.shape[0], targets_batch):
                cur_targets = targets[j:j+targets_batch]
                K, B, fit_counts, fit_loss = ransac_all_pairs(X, cur_targets, 
                                                            iters=iters, 
                                                            threshold=0.01,
                                                            min_inliers=10, 
                                                            sample_size=sample_size,
                                                            torch_gen=self.torch_gen) #optimize_kb(X, targets)
                all_Ks.append(K)
                all_Bs.append(B)
                all_fit_counts.append(fit_counts)
                all_fit_loss.append(fit_loss)
                # X_ = K.unsqueeze(-1) * X.unsqueeze(1) + B.unsqueeze(-1) # (n, k, dims) outputs 
                # del X
                # loss_per_term_per_test = self.fitness.get_loss(X_, custom_target=targets) # (n, k, dims) losses 
                # del X_
                # loss_per_term = loss_per_term_per_test.mean(dim=-1) # (n, k) losses
                # del loss_per_term_per_test
            if len(all_Ks) == 0:
                print(f"Zero K: Batch size {targets_batch}, {targets.shape[0]} samples, {len(cur_candidates)} candidates.")
                continue
            K = torch.cat(all_Ks, dim=1)
            B = torch.cat(all_Bs, dim=1)
            fit_counts = torch.cat(all_fit_counts, dim=1)
            fit_loss = torch.cat(all_fit_loss, dim=1)
            best_fit_id = fit_counts.argmax(dim=1) # (n,) min loss per term                
            for i, term in enumerate(cur_candidates):
                j = best_fit_id[i].item()
                term_fit_count = fit_counts[i,j].item()
                fl = fit_loss[i,j].item()           
                new_terms.append(((-term_fit_count, fl), K[i,j].item(), term, B[i,j].item()))

            del X, K, B, fit_counts, fit_loss
            for K in all_Ks:
                del K
            for B in all_Bs:
                del B
            for fit_counts in all_fit_counts:
                del fit_counts
            for fit_loss in all_fit_loss:
                del fit_loss
            pass
        res =  sorted(new_terms, key=lambda x: x[0])
        return res    