''' 
Implementation of spatial indices for efficient approximate nearest neighbor search (in amortized sense).

Idea of spacial indices is to avoid full search. 
For vector x and all semantix X of size n, full search would
require O(n) comparisons of k vector values. 
'''

from .base import VectorStorage, SpatialIndex
from .bin import BinIndex
from .grid import GridIndex
from .rtree import RTreeIndex
from .spearman import SpearmanCorIndex
from .cos import RCosIndex
from .inter import InteractionIndex
from .term_spatial import TermVectorStorage

__all__ = ['VectorStorage', 'SpatialIndex', 'TermVectorStorage',
           'BinIndex', 'GridIndex', 'RTreeIndex', 'SpearmanCorIndex', 
           'RCosIndex', 'InteractionIndex'
           ]