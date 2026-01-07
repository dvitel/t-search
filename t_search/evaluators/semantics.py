

from typing import Callable, Literal, Optional
import torch

from t_search.base import ServiceBase
from t_search.evaluators.term_spatial import TermVectorStorage
from t_search.syntax.term import Term, Value, Variable
from t_search.utils import stack_rows

class Semantics(ServiceBase):

    def __init__(self, *,
        var_bindings: dict[str, torch.Tensor],
        storage: TermVectorStorage,
        dims: int
        ):
        self.var_bindings = var_bindings
        self.storage = storage
        self.dims = dims

    def get_missing(self, terms: list[Term] | Term) -> list[Term]:
        if isinstance(terms, Term):
            terms = [terms]
        missing_terms = [t for t in terms if self.storage.get_semantics_for_term(t) is None]
        return missing_terms

    def get_outputs(self, terms: list[Term] | Term, return_type: Literal["list", "tensor"] = "list") -> list[torch.Tensor] | torch.Tensor | None:
        if isinstance(terms, Term):
            return self.get_binding(terms)
        selected_outputs = []
        for t in terms:
            t_outputs = self.get_binding(t)
            if t_outputs is None:
                raise ValueError(f"Term {t} semantics not found")
            selected_outputs.append(t_outputs)
        if return_type == "tensor":
            return stack_rows(selected_outputs, self.dims)
        return selected_outputs

    def get_binding(self, term: Term) -> Optional[torch.Tensor]:
        if isinstance(term, Variable):
            return self.var_bindings[term.var_id]
        if isinstance(term, Value):
            # return self.const_binding[term.value]
            return term.value  
        
        term_semantics = self.storage.get_semantics_for_term(term)        
        return term_semantics
    
    def is_const(
        self,
        outputs: torch.Tensor) -> Optional[float] | list[Optional[float]]:
        """Check if any of outputs is const or very slow function,
            outputs is 1d --> returns Optional mean
            outputs is 2d --> returns list of Optional means
            Returns mask 1d batch_id, and means 
        """
        if outputs.numel() == 1:
            return outputs.item()
        if outputs.dim() == 1:
            mean = outputs.mean()
            if torch.allclose(outputs, mean, atol=self.storage.index.atol, rtol=self.storage.index.rtol):
                return mean.item()
            else:
                return None
        if outputs.dim() == 2: # batch_id, dims
            mean = outputs.mean(dim=-1, keepdim=True) # batch_id, 1
            const_mask = torch.all(torch.isclose(outputs, mean, atol=self.storage.index.atol, rtol=self.storage.index.rtol), dim=-1) # batch_id
            res = [mean[i,0].item() if const_mask[i] else None for i in range(outputs.shape[0])]
            del mean, const_mask
            return res
        raise ValueError("Outputs should be 1d or 2d tensor.")
    
    def is_valid(self, term: Term) -> bool:
        ''' All outputs are non-infinite and non-nan '''
        return not self.storage.is_invalid(term)
              
    def set_binding(self, terms: list[Term] | Term, 
                          semantics: torch.Tensor) -> None:
        if isinstance(terms, Term):
            self.storage.insert([terms], semantics.unsqueeze(0))
        elif len(terms) > 0:
            self.storage.insert(terms, semantics)
        return
    
    def copy_binding(self, from_term: Term, to_term: Term) -> None:
        from_sem = self.storage.get_semantics_for_term(from_term)
        if from_sem is not None:
            self.storage.insert([to_term], from_sem.unsqueeze(0))
        return        

    def get_repr_terms(self) -> list[Term]:
        repr_terms = self.storage.get_repr_terms()
        return repr_terms

    def get_iter_metrics(self):
        iter_num_semantics = self.storage.num_sem()
        iter_num_sem_terms = self.storage.num_terms()
        iter_num_invalid_terms = len(self.storage.invalid_terms)
        return {
            'iter_num_semantics': [iter_num_semantics],
            'iter_num_sem_terms': [iter_num_sem_terms],
            "iter_num_invalid_terms": [iter_num_invalid_terms]
        }