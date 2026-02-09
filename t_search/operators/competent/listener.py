
from typing import Callable

import torch
from t_search.base import ServiceBase
from t_search.evaluators.semantics import Semantics
from t_search.evaluators.term_spatial import TermVectorStorage
from t_search.operators.classic.up2d import Up2D
from t_search.operators.competent.utils import DesiredSemantics, InversionCache, get_desired_semantics
from t_search.operators.initialization import Initialization
from t_search.syntax import Term
from t_search.syntax.syntax import Syntax
from t_search.utils import stack_rows
from ..listeners import EvalListener

class CompetentListener(ServiceBase, EvalListener):
    ''' Listener that is used by competent operators CM, CX
        May dynamically register programs in the library. 
        Performs initialization of the library. Injected by competent operators.
    '''

    def __init__(self, *,
                    syntax: Syntax,
                    semantics: Semantics,
                    target: torch.Tensor, 
                    init_op: Initialization, 
                    term_vector_storage: TermVectorStorage,
                    # index_init_depth: int | None = None, 
                    dynamic_index: bool = False,
                    index_max_size: int = 1e10,
                 ):
        self.syntax = syntax
        self.semantics = semantics
        self.index = term_vector_storage # used as library of semantics 
        # self.index_init_depth = index_init_depth # if None, dynamic library - uses any available term. 
        self.dynamic_index = dynamic_index
        self.index_max_size = index_max_size
        self.inv_cache = InversionCache()
        self.desired_target: DesiredSemantics | None = None
        self.target: torch.Tensor = target
        self.desired_target = get_desired_semantics(target)
        self.init_op = init_op #Up2D(syntax = self.syntax, depth = self.index_init_depth)
        self.inited: bool = False

    def init(self, *, evaluator, **services):
        if not self.inited:
            lib_terms = self.init_op()
            evaluator.eval(lib_terms)
            if self.term_vector_storage is self.semantics.storage:
                return
            valid_terms = [t for t in lib_terms if self.semantics.is_valid(t)]
            semantics = self.semantics.get_outputs(valid_terms, return_type="tensor")
            self.index.insert(valid_terms, semantics) 
            del semantics
        self.inited = True

    def on_eval(self, terms: list[Term], semantics: list[torch.Tensor]):
        if self.term_vector_storage is self.semantics.storage:
            return
        if self.inited and self.dynamic_index and self.index.num_terms() < self.index_max_size:
            vectors = stack_rows(semantics, self.semantics.dims)
            self.index.insert(terms, vectors)
            del vectors
    
    def get_desired_semantics(self, term: Term, semantics: torch.Tensor) -> DesiredSemantics:
        if term not in self.inv_cache.term_semantics:
            self.inv_cache.term_semantics[term] = get_desired_semantics(semantics)
        return self.inv_cache.term_semantics[term]
    
    def get_desired_target(self):
        if self.desired_target is None:
            raise ValueError("Desired target semantics not initialized.")
        return self.desired_target
