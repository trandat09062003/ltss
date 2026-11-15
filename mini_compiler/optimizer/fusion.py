from mini_compiler.core.ops import Op
from mini_compiler.core.tensor import Tensor

class FusedOp(Op):
    """
    Đại diện cho một cụm các phép toán (subgraph) được gộp lại (Fused Kernel).
    Thay vì chạy từng kernel riêng biệt, FusedOp sẽ chạy 1 kernel CUDA duy nhất.
    """
    op_type = "fused_pointwise"
    is_fusible = False  # Đã gộp rồi, không gộp lồng nữa

    def __init__(self, sub_nodes, inputs, outputs, name=None):
        super().__init__(inputs, name=name or f"fused_kernel_{id(self) % 10000}")
        self.sub_nodes = sub_nodes
        self.outputs = outputs
        self._compiled_kernel = None

    def __repr__(self):
        ops_str = " -> ".join([n.op_type for n in self.sub_nodes])
        return f"FusedOp({self.name}: [{ops_str}])"


class FusionOptimizer:
    """
    Tối ưu hóa Đồ thị tính toán bằng cách gom cụm (Kernel Fusion).
    Nguyên lý: Gom các chuỗi toán tử Pointwise/Element-wise liên tiếp
    (như Add, Mul, ReLU, Dropout) thành một FusedOp duy nhất.
    """
    def __init__(self, graph):
        self.graph = graph

    def run_fusion(self):
        """
        Thực hiện pass gom cụm trên đồ thị DAG.
        Trả về danh sách các Node sau khi đã tối ưu (gồm FusedOp và các Standalone Op).
        """
        nodes = self.graph.nodes
        if not nodes:
            return []

        # Phân tích mức sử dụng của từng tensor để xác định tensor nào cần lưu ra ngoài
        tensor_consumer_count = {}
        for node in nodes:
            for inp in node.inputs:
                tensor_consumer_count[inp] = tensor_consumer_count.get(inp, 0) + 1

        optimized_nodes = []
        current_cluster = []

        def flush_cluster():
            nonlocal current_cluster
            if not current_cluster:
                return

            if len(current_cluster) == 1:
                # Cụm chỉ có 1 node, giữ nguyên không cần gộp
                optimized_nodes.append(current_cluster[0])
            else:
                # Gom cụm thành FusedOp
                cluster_nodes = list(current_cluster)
                cluster_node_set = set(cluster_nodes)

                # Tìm inputs ngoài cụm
                ext_inputs = []
                seen_inputs = set()
                for node in cluster_nodes:
                    for inp in node.inputs:
                        # Nếu input không do node nào trong cụm tạo ra
                        if inp.creator not in cluster_node_set:
                            if inp not in seen_inputs:
                                seen_inputs.add(inp)
                                ext_inputs.append(inp)

                # Tìm outputs cần xuất ra ngoài cụm (được dùng ở node ngoài hoặc là root output)
                ext_outputs = []
                root_set = set(self.graph.roots)
                for node in cluster_nodes:
                    for out in node.outputs:
                        if out in root_set or tensor_consumer_count.get(out, 0) > sum(1 for n in cluster_nodes if out in n.inputs):
                            ext_outputs.append(out)

                # Nếu không có output nào được đánh dấu (chỉ dùng nội bộ), lấy output của node cuối
                if not ext_outputs:
                    ext_outputs = list(cluster_nodes[-1].outputs)

                fused_node = FusedOp(
                    sub_nodes=cluster_nodes,
                    inputs=ext_inputs,
                    outputs=ext_outputs,
                    name=f"fused_{'_'.join(n.op_type for n in cluster_nodes)}_{id(cluster_nodes[0]) % 1000}"
                )
                optimized_nodes.append(fused_node)

            current_cluster = []

        # Quét tuần tự đồ thị theo thứ tự tô-pô
        for node in nodes:
            if getattr(node, 'is_fusible', False):
                # Node có thể gom cụm
                if not current_cluster:
                    current_cluster.append(node)
                else:
                    # Kiểm tra xem node này có nối tiếp / tương thích với cluster hiện tại không
                    # Trong bài toán elementwise: nếu cùng shape, gom tiếp
                    current_cluster.append(node)
            else:
                # Gặp node không thể gộp (e.g. MatMul) -> flush cluster hiện tại ra
                flush_cluster()
                optimized_nodes.append(node)

        flush_cluster()
        return optimized_nodes

