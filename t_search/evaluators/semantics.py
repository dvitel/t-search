

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
        add_metrics: Callable,
        dims: int
        ):
        self.var_bindings = var_bindings
        self.storage = storage
        self.invalid_terms: dict[Term, torch.Tensor] = {}
        self.add_metrics = add_metrics
        self.dims = dims

    def get_missing(self, terms: list[Term] | Term) -> list[Term]:
        if isinstance(terms, Term):
            terms = [terms]
        missing_terms = [t for t in terms if self.storage.get_semantics_for_term(t) is None and t not in self.invalid_terms]
        return missing_terms

    def get_outputs(self, terms: list[Term] | Term, return_type: Literal["list", "tensor"] = "list") -> list[torch.Tensor] | torch.Tensor | None:
        if isinstance(terms, Term):
            if terms in self.invalid_terms:
                return self.invalid_terms[terms]
            return self.storage.get_semantics_for_term(terms)
        selected_outputs = []
        for t in terms:
            t_outputs = self.invalid_terms[t] if t in self.invalid_terms else self.storage.get_semantics_for_term(terms)
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
        if term in self.invalid_terms:
            return self.invalid_terms[term]
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
        return term not in self.invalid_terms    
              
    def set_binding(self, valid_terms: list[Term], 
                          valid_semantics: torch.Tensor,
                          invalid_terms: list[Term], 
                          invalid_outputs: torch.Tensor) -> None:
        if len(valid_terms) > 0:
            self.storage.insert(valid_terms, valid_semantics)

        for term, outputs in zip(invalid_terms, invalid_outputs):
            self.invalid_terms[term] = outputs
        return
    
    def get_finalizer(self):
        self.add_metrics(invalid_terms=len(self.invalid_terms))

    def get_repr_terms(self) -> list[Term]:
        repr_terms = self.storage.get_repr_terms()
        return repr_terms
