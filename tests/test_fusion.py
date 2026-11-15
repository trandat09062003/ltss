import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_compiler.core import Tensor, ComputationalGraph
from mini_compiler.optimizer import FusionOptimizer

def test_fusion():
    x = Tensor([1.0, 2.0], name='x')
    w = Tensor([3.0, 4.0], name='w')
    b = Tensor([0.5, 0.5], name='b')
    
    # Chuỗi: x * w + b -> relu -> dropout
    y = (x * w + b).relu().dropout(p=0.2)
    
    g = ComputationalGraph(y)
    print("=== Trước khi Fusion ===")
    print(f"Tổng số node ban đầu: {len(g.nodes)}")
    g.print_graph()
    
    opt = FusionOptimizer(g)
    opt_nodes = opt.run_fusion()
    
    print("\n=== Sau khi Fusion ===")
    print(f"Tổng số node sau tối ưu: {len(opt_nodes)}")
    for n in opt_nodes:
        print(f"Node: {n}")
        print(f"  Inputs: {[i.name for i in n.inputs]}")
        print(f"  Outputs: {[o.name for o in n.outputs]}")

if __name__ == "__main__":
    test_fusion()
