import math
from mini_compiler.core.graph import ComputationalGraph
from mini_compiler.optimizer.fusion import FusionOptimizer, FusedOp
from mini_compiler.codegen.cuda_codegen import CUDACodeGenerator
from mini_compiler.runtime.jit_nvrtc import NVRTCCompiler, CUDAExecutor, HAS_CUDA

if HAS_CUDA:
    import cupy as cp
else:
    import numpy as np


class CompiledProgram:
    """
    Chương trình đã được biên dịch hoàn chỉnh.
    Chứa đồ thị tối ưu, các kernel CUDA đã JIT compile và hàm execute.
    """
    def __init__(self, optimized_nodes, kernels, executor):
        self.optimized_nodes = optimized_nodes
        self.kernels = kernels  # Dict[op_id, (kernel_func, kernel_name, cuda_source)]
        self.executor = executor

    def execute(self, feed_dict, block_size=256):
        """
        Thực thi forward pass trên dữ liệu đầu vào.
        feed_dict: Dict[Tensor, array/buffer]
        """
        # Map tensor -> buffer
        tensor_buffers = dict(feed_dict)

        for node in self.optimized_nodes:
            if isinstance(node, FusedOp):
                # Lấy dữ liệu input buffers
                in_buffers = [tensor_buffers[inp] for inp in node.inputs]
                
                # Cấp phát output buffer
                N = in_buffers[0].size
                grid_size = math.ceil(N / block_size)

                if HAS_CUDA:
                    out_buffers = [cp.empty_like(in_buffers[0]) for _ in node.outputs]
                    kernel_func, _, _ = self.kernels[id(node)]
                    
                    # Chuẩn bị danh sách tham số: [in_0, in_1, ..., out_0, N]
                    args = tuple(in_buffers + out_buffers + [N])
                    self.executor.launch(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), args)
                else:
                    # CPU fallback cho test logic
                    out_buffers = [np.empty_like(in_buffers[0]) for _ in node.outputs]

                for out_t, buf in zip(node.outputs, out_buffers):
                    tensor_buffers[out_t] = buf
            else:
                # Standalone node (ví dụ MatMul)
                if HAS_CUDA:
                    # Dùng CuPy cuBLAS cho MatMul
                    res = node.inputs[0].data @ node.inputs[1].data
                else:
                    res = node.inputs[0].data @ node.inputs[1].data
                tensor_buffers[node.outputs[0]] = res

        return tensor_buffers


class MiniCompiler:
    """
    Trình biên dịch Mini Compiler cho Deep Learning.
    Quy trình:
      Graph Tracing -> Fusion Pass -> CUDA Codegen -> NVRTC JIT -> Compiled Program
    """
    def __init__(self):
        self.codegen = CUDACodeGenerator()
        self.compiler = NVRTCCompiler()
        self.executor = CUDAExecutor(self.compiler)

    def compile(self, root_tensors):
        """
        Biên dịch đồ thị bắt đầu từ các root tensors.
        """
        # 1. Xây dựng đồ thị tính toán ban đầu
        graph = ComputationalGraph(root_tensors)
        
        # 2. Chạy Fusion Pass (Gom cụm các phép toán pointwise)
        optimizer = FusionOptimizer(graph)
        optimized_nodes = optimizer.run_fusion()

        # 3. Sinh mã và JIT Compile cho các FusedOp
        kernels = {}
        for node in optimized_nodes:
            if isinstance(node, FusedOp):
                k_name, source = self.codegen.generate_fused_kernel(node)
                kernel_func = self.compiler.compile(source, k_name)
                kernels[id(node)] = (kernel_func, k_name, source)

        return CompiledProgram(optimized_nodes, kernels, self.executor)

