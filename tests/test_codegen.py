import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_compiler.core import Tensor, ComputationalGraph
from mini_compiler.optimizer import FusionOptimizer
from mini_compiler.codegen import CUDACodeGenerator

def test_codegen():
    x = Tensor([1.0, 2.0], name='x')
    w = Tensor([3.0, 4.0], name='w')
    b = Tensor([0.5, 0.5], name='b')
    
    # Chuỗi: x * w + b -> relu -> dropout
    y = (x * w + b).relu().dropout(p=0.2)
    
    g = ComputationalGraph(y)
    opt = FusionOptimizer(g)
    opt_nodes = opt.run_fusion()
    
    fused_node = opt_nodes[0]
    codegen = CUDACodeGenerator()
    kernel_name, cuda_code = codegen.generate_fused_kernel(fused_node)
    
    print(f"Generated Kernel Name: {kernel_name}")
    print("=== Generated CUDA C++ Code ===")
    print(cuda_code)
    print("================================")

if __name__ == "__main__":
    test_codegen()

