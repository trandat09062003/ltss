# Mini Compiler cho Deep Learning: Autograd + Kernel Fusion
> **Môn học:** Lập Trình Song Song (Parallel Programming) - HUST  
> **Đề tài 3:** Mini compiler cho Deep Learning: autograd + kernel fusion  
> **Báo cáo chi tiết & Số liệu thực nghiệm:** Xem [BAO_CAO_KET_QUA.md](BAO_CAO_KET_QUA.md)

---

## 📌 1. Mục tiêu Đề tài
- Tự viết engine tự động đạo hàm (**Reverse-mode Autograd**), xây dựng đồ thị tính toán (**Computational Graph / DAG**).
- Tự động nhận diện và gom cụm chuỗi phép toán liên tiếp (ví dụ: `x * w + b -> ReLU -> Dropout`) thành một kernel duy nhất (**Kernel Fusion**).
- Tự động sinh mã nguồn **CUDA C++** lúc chạy (JIT) và biên dịch nạp vào GPU bằng **NVRTC** (*NVIDIA Runtime Compilation*).
- Giải thích và chứng minh định lượng cơ chế tối ưu: **Kernel Fusion giúp loại bỏ việc đọc/ghi mảng trung gian vào Global Memory (VRAM), tận dụng tốc độ của Registers**.

---

## 🏗️ 2. Cấu trúc Thư mục Dự án

```text
ltss/
│
├── mini_compiler/               # Mã nguồn chính của Mini Compiler
│   ├── core/                    # Computational Graph & Autograd Engine
│   │   ├── tensor.py            # Lớp Tensor, reverse-mode autodiff, topo sort
│   │   ├── ops.py               # Các toán tử: Add, Mul, ReLU, Dropout, MatMul
│   │   └── graph.py             # Biểu diễn DAG, debug và sinh Mermaid diagram
│   │
│   ├── optimizer/               # Mid-end Graph Optimizer
│   │   └── fusion.py            # Thuật toán gom cụm (Kernel Fusion pass)
│   │
│   ├── codegen/                 # Backend Code Generator
│   │   └── cuda_codegen.py      # Sinh mã CUDA C++ cho kernel fused & single op
│   │
│   ├── runtime/                 # JIT Runtime Execution
│   │   └── jit_nvrtc.py         # NVRTC compiler, CUDA Events timer, CPU fallback
│   │
│   └── compiler.py              # Pipeline tích hợp hoàn chỉnh (MiniCompiler)
│
├── benchmark/                   # Script đo đạc & phân tích hiệu năng
│   └── benchmark_fusion.py      # So sánh Non-fused vs Fused vs PyTorch
│
├── tests/                       # Unit tests kiểm thử từng thành phần
│   ├── test_fusion.py           # Kiểm tra pass gom cụm đồ thị
│   ├── test_codegen.py          # Kiểm tra sinh mã CUDA C++
│   └── test_compiler.py         # Kiểm tra end-to-end pipeline
│
├── notebooks/                   # Jupyter Notebook chạy trực tiếp trên GPU
│   └── mini_compiler_colab.ipynb # Notebook chạy trực tiếp trên JupyterLab / Colab
│
├── BAO_CAO_KET_QUA.md           # Báo cáo thực nghiệm chi tiết kèm số liệu đo đạc
└── ket_qua_benchmark.txt        # Kết quả benchmark thực tế trên RTX 3060
```

---

## 🚀 3. Hướng dẫn Chạy Thử nghiệm

### Cách 1: Chạy trực tiếp trên JupyterLab / Google Colab (Có GPU NVIDIA)
1. Mở notebook: [`notebooks/mini_compiler_colab.ipynb`](notebooks/mini_compiler_colab.ipynb).
2. Chạy toàn bộ các cells để quan sát:
   - Xây dựng đồ thị & Autograd.
   - Gộp chuỗi `x * w + b -> ReLU -> Dropout`.
   - Sinh mã CUDA C++ và JIT compile qua NVRTC.
   - Đo benchmark thời gian thực tế và vẽ biểu đồ so sánh với PyTorch.

Hoặc chạy trực tiếp trên Terminal của GPU server:
```bash
python benchmark/benchmark_fusion.py | tee ket_qua_benchmark.txt
```

### Cách 2: Chạy kiểm thử trên máy Local (CPU Fallback Mode)
Hệ thống tích hợp sẵn cơ chế **CPU Fallback** để kiểm tra logic tính toán mà không cần card NVIDIA:

```bash
# Kiểm tra thuật toán Fusion
python tests/test_fusion.py

# Kiểm tra bộ sinh mã CUDA C++
python tests/test_codegen.py

# Kiểm tra toàn bộ compiler pipeline
python tests/test_compiler.py

# Xem phân tích lý thuyết lưu lượng VRAM
python benchmark/benchmark_fusion.py
```

---

## 📊 4. Kết Quả Thực Nghiệm Trên GPU NVIDIA GeForce RTX 3060

Số liệu đo đạc thực tế với chuỗi phép toán $N = 10,000,000$ phần tử:
$$Y = \text{Dropout}(\text{ReLU}(X \cdot W + B))$$

```text
==================== KẾT QUẢ ĐO ĐẠC THỰC TẾ ====================
1. Non-Fused Latency :   1.2084 ms | Traffic: 381.5 MB | Achieved BW: 331.02 GB/s
2. Fused Latency     :   0.4765 ms | Traffic: 152.6 MB | Achieved BW: 335.77 GB/s
----------------------------------------------------------------
==> TỐC ĐỘ TĂNG (Speedup): 2.54x (Nhanh hơn gấp 2.54 lần!)
==> TIẾT KIỆM LƯU LƯỢNG  : 60.0% VRAM bandwidth
================================================================
```

### Vì sao Kernel Fusion nhanh hơn?
* **Non-fused (4 kernels rời rạc):** Phải lưu 3 mảng trung gian vào VRAM $\rightarrow$ Tổng đọc/ghi: $(6N + 4N) \times 4\text{ bytes} = 40N\text{ bytes} = \mathbf{381.5\text{ MB}}$.
* **Fused Kernel:** Toàn bộ biến trung gian nằm trên **Registers** $\rightarrow$ Chỉ đọc $X, W, B$ một lần và ghi $Y$ một lần $\rightarrow$ Tổng đọc/ghi: $(3N + 1N) \times 4\text{ bytes} = 16N\text{ bytes} = \mathbf{152.6\text{ MB}}$.
* **Tiết kiệm chính xác 60.0% lưu lượng VRAM**, đồng thời đạt **$335.77\text{ GB/s}$** ($>93\%$ giới hạn vật lý của RTX 3060).
