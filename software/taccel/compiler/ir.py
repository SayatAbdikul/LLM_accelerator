"""Intermediate Representation for the compiler."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class IRNode:
    """A single operation in the IR graph."""
    op: str  # e.g., "matmul", "layernorm", "softmax", "gelu", "vadd", "reshape", etc.
    name: str  # unique identifier
    inputs: List[str] = field(default_factory=list)  # names of input IRNodes or weight tensors
    output_shape: Tuple[int, ...] = ()
    attrs: Dict[str, Any] = field(default_factory=dict)
    # Scale info populated by quantizer
    output_scale: Optional[float] = None
    weight_name: Optional[str] = None  # associated weight tensor name


class IRGraph:
    """Ordered list of IR nodes representing the computation."""

    def __init__(self):
        self.nodes: List[IRNode] = []
        self.node_map: Dict[str, IRNode] = {}

    def add_node(self, node: IRNode) -> IRNode:
        self.nodes.append(node)
        self.node_map[node.name] = node
        return node

    def get_node(self, name: str) -> Optional[IRNode]:
        return self.node_map.get(name)

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def compute_last_uses(self) -> Dict[str, int]:
        """Return {node_name → index of last node that uses it as input}.

        Only covers node outputs (not weight names). Used by codegen to know
        when an ABUF allocation can be freed.
        """
        node_names = {n.name for n in self.nodes}
        last_use: Dict[str, int] = {}
        for idx, node in enumerate(self.nodes):
            for inp in node.inputs:
                if inp in node_names:
                    last_use[inp] = idx
        return last_use
