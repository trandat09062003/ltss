# Mini Compiler cho Deep Learning: Autograd Engine & Kernel Fusion

Đồ án môn học **Lập trình song song** – Trường Công nghệ Thông tin & Truyền thông, Đại học Bách Khoa Hà Nội (HUST).

Hệ thống hiện thực một trình biên dịch thu nhỏ (Mini Compiler) cho Deep Learning, mô phỏng lại nguyên lý cốt lõi của các trình biên dịch công nghiệp như **PyTorch TorchInductor**, **OpenAI Triton** và **Google XLA**:
1. Xây dựng đồ thị tính toán có hướng (DAG) và động cơ tự động tính đạo hàm ngược (Reverse-mode Automatic Differentiation).
2. Tự động gom cụm các chuỗi toán tử xử lý theo từng phần tử liên tiếp (ví dụ: $X \cdot W + B \to \text{ReLU} \to \text{Dropout}$) thành một kernel duy nhất (**Kernel Fusion**).
3. Tự động sinh mã nguồn **CUDA C++** lúc chạy (JIT) và nạp trực tiếp vào GPU thông qua **NVRTC** (*NVIDIA Runtime Compilation*).
4. Phân tích định lượng và chứng minh thực nghiệm nguyên nhân tăng tốc: **Loại bỏ việc đọc/ghi mảng trung gian qua VRAM, chuyển toàn bộ tính toán trung gian lên thanh ghi (Registers)**.

> 📄 **Báo cáo kỹ thuật chi tiết:** Xem tại [BAO_CAO_KET_QUA.md](BAO_CAO_KET_QUA.md).

---

## 1. Kết Quả Đo Đạc Thực Tế Trên GPU NVIDIA GeForce RTX 3060

Kiểm thử với tensor có kích thước $N = 10,000,000$ phần tử kiểu `float32` (tổng dung lượng mỗi tensor $40\text{ MB}$):
$$Y = \text{Dropout}(\text{ReLU}(X \cdot W + B))$$

```text
==================== KẾT QUẢ ĐO ĐẠC THỰC TẾ ====================
1. Non-Fused Latency :   1.2084 ms | Traffic: 381.5 MB | Achieved BW: 331.02 GB/s
2. Fused Latency     :   0.4765 ms | Traffic: 152.6 MB | Achieved BW: 335.77 GB/s
----------------------------------------------------------------
==> TỐC ĐỘ TĂNG (Speedup): 2.54x (Nhanh hơn gấp 2.54 lần)
==> TIẾT KIỆM LƯU LƯỢNG  : 60.0% VRAM bandwidth
================================================================
```

* **Băng thông bộ nhớ:** Cả hai kernel đều khai thác **$> 93\%$ băng thông cực đại của RTX 3060** ($335.77\text{ GB/s}$ so với mức trần $360\text{ GB/s}$).
* **Bản chất tối ưu:** Vì bài toán bị nghẽn ở băng thông bộ nhớ (Memory-bound), khi Fused Kernel cắt giảm lượng đọc/ghi từ $381.5\text{ MB}$ xuống $152.6\text{ MB}$ (giảm $2.5\times$), thời gian chạy giảm tương ứng từ $1.2084\text{ ms}$ xuống $0.4765\text{ ms}$ ($2.54\times$).

---

## 2. Cấu Trúc Mã Nguồn

```text
ltss/
├── mini_compiler/               # Mã nguồn chính của Mini Compiler
│   ├── core/                    # Computational Graph & Autograd Engine
│   │   ├── tensor.py            # Lớp Tensor, topological sort, lan truyền gradient
│   │   ├── ops.py               # Các toán tử nguyên thủy: Add, Mul, ReLU, Dropout, MatMul
│   │   └── graph.py             # Quản lý đồ thị tính toán và xuất cấu trúc DAG
│   │
│   ├── optimizer/               # Mid-end Graph Optimizer
│   │   └── fusion.py            # Thuật toán gom cụm (Kernel Fusion pass)
│   │
│   ├── codegen/                 # Backend Code Generator
│   │   └── cuda_codegen.py      # Bộ sinh mã CUDA C++ tận dụng thanh ghi và in-thread PRNG
│   │
│   ├── runtime/                 # JIT Runtime Execution
│   │   └── jit_nvrtc.py         # NVRTC JIT compilation, caching, CUDA Events timer
│   │
│   └── compiler.py              # Pipeline tích hợp toàn diện (MiniCompiler)
│
├── benchmark/                   # Module đo đạc hiệu năng
│   └── benchmark_fusion.py      # Benchmark Non-fused vs Fused với CUDA Events
│
├── tests/                       # Unit tests kiểm tra tính đúng đắn
│   ├── test_fusion.py           # Kiểm tra pass gom cụm trên đồ thị
│   ├── test_codegen.py          # Kiểm tra mã nguồn CUDA C++ sinh ra
│   └── test_compiler.py         # Kiểm tra end-to-end pipeline
│
├── notebooks/                   # Jupyter Notebook trực quan
│   └── mini_compiler_colab.ipynb # Notebook chạy trực tiếp trên GPU Server / Google Colab
│
├── BAO_CAO_KET_QUA.md           # Báo cáo kỹ thuật chi tiết
└── ket_qua_benchmark.txt        # Kết quả benchmark log từ GPU RTX 3060
```

---

## 3. Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### Chạy trên Server có GPU NVIDIA (JupyterLab / Linux):
Yêu cầu môi trường đã có Driver NVIDIA và Python $\ge 3.8$:

```bash
# 1. Cài đặt các thư viện phụ trợ
pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 matplotlib pandas

# 2. Thiết lập đường dẫn thư viện NVRTC
export LD_LIBRARY_PATH=$(python -c "import os, nvidia.cuda_nvrtc; print(os.path.dirname(nvidia.cuda_nvrtc.__file__) + '/lib')"):$LD_LIBRARY_PATH

# 3. Chạy benchmark đo đạc và lưu kết quả
python benchmark/benchmark_fusion.py | tee ket_qua_benchmark.txt
```

### Chạy trên máy cá nhân không có GPU (CPU Fallback Mode):
Hệ thống tích hợp sẵn cơ chế giả lập CPU để kiểm tra tính đúng đắn của giải thuật:

```bash
# Kiểm tra pass gom cụm đồ thị
python tests/test_fusion.py

# Kiểm tra bộ sinh mã CUDA C++
python tests/test_codegen.py

# Kiểm tra pipeline biên dịch
python tests/test_compiler.py
```
