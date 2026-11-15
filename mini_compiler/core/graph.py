from mini_compiler.core.tensor import Tensor
from mini_compiler.core.ops import Op

class ComputationalGraph:
    """
    Biểu diễn Đồ thị tính toán (DAG) gồm các Tensor (Edges) và Op (Nodes).
    Hỗ trợ duyệt topo, debug, visualize và chuẩn bị cho pass tối ưu hóa.
    """
    def __init__(self, root_tensors):
        if isinstance(root_tensors, Tensor):
            root_tensors = [root_tensors]
        self.roots = root_tensors
        self.nodes = []      # Danh sách Op theo thứ tự topo
        self.tensors = []    # Danh sách Tensor
        self.build_graph()

    def build_graph(self):
        self.nodes = []
        self.tensors = []
        visited_tensors = set()
        visited_nodes = set()

        def dfs(tensor):
            if tensor in visited_tensors:
                return
            visited_tensors.add(tensor)
            self.tensors.append(tensor)

            if tensor.creator is not None:
                op = tensor.creator
                for inp in op.inputs:
                    dfs(inp)
                if op not in visited_nodes:
                    visited_nodes.add(op)
                    self.nodes.append(op)

        for root in self.roots:
            dfs(root)

    def print_graph(self):
        print("=== Computational Graph (Topological Order) ===")
        for i, node in enumerate(self.nodes):
            in_names = [inp.name for inp in node.inputs]
            out_names = [out.name for out in node.outputs]
            fusible_tag = "[FUSIBLE]" if getattr(node, 'is_fusible', False) else "[STANDALONE]"
            print(f"[{i:02d}] {fusible_tag:<12} {node.name} ({node.op_type}): ({', '.join(in_names)}) -> ({', '.join(out_names)})")
        print("===============================================")

    def to_mermaid(self):
        """Sinh mã Mermaid diagram để trực quan hóa đồ thị."""
        lines = ["flowchart TD"]
        for node in self.nodes:
            fusible = getattr(node, 'is_fusible', False)
            style = ":::fusible" if fusible else ":::standalone"
            lines.append(f'    {node.name}["{node.op_type} ({node.name})"]{style}')
            for inp in node.inputs:
                lines.append(f'    {inp.name} --> {node.name}')
            for out in node.outputs:
                lines.append(f'    {node.name} --> {out.name}')
        lines.append("    classDef fusible fill:#a8f0a5,stroke:#2e7d32,stroke-width:2px;")
        lines.append("    classDef standalone fill:#bbdefb,stroke:#1565c0,stroke-width:2px;")
        return "\n".join(lines)

