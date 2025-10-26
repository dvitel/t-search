
from typing import Generator, Optional

import numpy as np
from .term import TermPos


def shuffle_positions(positions: list[TermPos],
                        select_node_leaf_prob: Optional[float] = 0.1,
                        rnd: np.random.RandomState = np.random) -> np.ndarray:
    pos_proba = rnd.rand(len(positions))
    if select_node_leaf_prob is not None:
        proba_mod = np.array([select_node_leaf_prob if pos.term.arity() == 0 else (1 - select_node_leaf_prob) for pos in positions ], dtype=float)
        pos_proba *= proba_mod
    pos_proba = 1 - pos_proba
    return np.argsort(pos_proba)



def shuffled_position_flow(positions: list[TermPos], leaf_proba: float | None = None, rnd: np.random.RandomState = np.random) -> Generator[TermPos]:
    if len(positions) == 0:
        return
    ordered_pos_ids = shuffle_positions(positions, 
                                    select_node_leaf_prob = leaf_proba, 
                                    rnd = rnd)        
    for pos_id in ordered_pos_ids:
        yield positions[pos_id]

def random_position_flow(positions: list[TermPos], rnd: np.random.RandomState = np.random) -> Generator[TermPos]:
    if len(positions) == 0:
        return
    
    while True:
        pos_id = rnd.randint(len(positions))
        yield positions[pos_id]