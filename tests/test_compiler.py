import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_compiler import Tensor, MiniCompiler

def test_compiler():
    print("Khởi tạo mạng nơ-ron đơn giản: x * w + b -> relu -> dropout...")
    x = Tensor([1.0, 2.0, 3.0, 4.0], name="x")
    w = Tensor([0.5, 0.5, 0.5, 0.5], name="w")
    b = Tensor([0.1, 0.1, 0.1, 0.1], name="b")
    
    y = (x * w + b).relu().dropout(p=0.0)
    
    print("Tiến hành biên dịch qua MiniCompiler...")
    compiler = MiniCompiler()
    program = compiler.compile(y)
    
    print(f"Số node sau khi tối ưu: {len(program.optimized_nodes)}")
    for node in program.optimized_nodes:
        print(f"  Node: {node}")
        if id(node) in program.kernels:
            _, k_name, src = program.kernels[id(node)]
            print(f"  Kernel: {k_name}")
            print("  --- CUDA C++ Source preview ---")
            for line in src.splitlines()[:12]:
                print(f"    {line}")
            print("    ...")

if __name__ == "__main__":
    test_compiler()

