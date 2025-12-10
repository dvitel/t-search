''' MCTS Tree abstraction on top of Terms 
    MCTree node contains a Term, edge represents the mutation operator applied to term.
    This differs from other applications, where node could be incomplete partial solutions that are simulated. 
'''

from dataclasses import dataclass, field
from typing import Callable

from t_search.operators.operator import Operator
from t_search.syntax.term import Term


@dataclass(frozen=False, eq=False, unsafe_hash=False, repr=False)
class MCTreeNode:
    term: Term
    operator: Operator # applied mutation onto term  
    Q: float = 0.0 # represents quality value for selection, in contrast to fitness 
    N: int = 0 # num visits ??? - check - can we use len(children) instead ?
    #- it is aggregated fitness that represents neighborhood search - averaged over children mutations 
    children: list['MCTreeNode'] = field(default_factory=list)
    parent: 'MCTreeNode | None' = None


class MCTree:
    ''' All terms are stored in this tree '''    

    def __init__(self):
        self.roots: list[MCTreeNode] = []


    def pick_node(self, selector: Callable) -> MCTreeNode:
        cur_nodes = self.roots
        while True:
            proceed, node = selector(cur_nodes)
            if not proceed or len(node.children) == 0:
                return node
            cur_nodes = node.children    

    def create_child(self, parent: MCTreeNode, child_term: Term, operator: Operator) -> MCTreeNode:
        child_node = MCTreeNode(term=child_term, operator=operator, parent=parent)
        parent.children.append(child_node)
        return child_node    

    def update_nodes(self, start_node: MCTreeNode, updater: Callable):
        ''' Updates all nodes in the tree using updater function '''
        cur_node = start_node
        while cur_node is not None:
            updater(cur_node)
            cur_node = cur_node.parent            