<div align="center">

# Mini Compiler cho Deep Learning: Autograd Engine & Kernel Fusion JIT

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NVRTC](https://img.shields.io/badge/JIT-NVRTC%20Compiler-76B900.svg)](https://docs.nvidia.com/cuda/nvrtc/index.html)
[![GPU](https://img.shields.io/badge/Tested%20on-RTX%203060%2012GB-orange.svg)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

</div>

---

## 📖 MỤC LỤC
1. [Giới thiệu tổng quan & Động lực nghiên cứu](#-1-giới-thiệu-tổng-quan--động-lực-nghiên-cứu)
2. [Bản chất bài toán: Memory-Bound & Roofline Model](#-2-bản-chất-bài-toán-memory-bound--roofline-model)
3. [Kiến trúc hệ thống Mini Compiler 4 tầng](#-3-kiến-trúc-hệ-thống-mini-compiler-4-tầng)
   - 3.1. Frontend & Reverse-mode Autograd DAG
   - 3.2. Mid-end Graph Optimizer: Thuật toán Kernel Fusion Pass
   - 3.3. Backend Codegen: Tối ưu thanh ghi & In-thread PRNG
   - 3.4. JIT Runtime qua NVRTC & CUDA Driver API
4. [Mã nguồn CUDA C++ tự sinh lúc chạy (JIT Generated Code)](#-4-mã-nguồn-cuda-c-tự-sinh-lúc-chạy-jit-generated-code)
5. [Engineering Log: 3 Thách thức kỹ thuật thực tế trên GPU Server](#-5-engineering-log-3-thách-thức-kỹ-thuật-thực-tế-trên-gpu-server)
6. [Kết quả đo đạc thực nghiệm trên NVIDIA RTX 3060](#-6-kết-quả-đo-đạc-thực-nghiệm-trên-nvidia-rtx-3060)
7. [Đối chiếu kiến trúc với TorchInductor, Triton và XLA](#-7-đối-chiếu-kiến-trúc-với-torchinductor-triton-và-xla)
8. [Cấu trúc thư mục & Hướng dẫn tái lập thực nghiệm](#-8-cấu-trúc-thư-mục--hướng-dẫn-tái-lập-thực-nghiệm)
9. [Thông tin nhóm thực hiện](#-9-thông-tin-nhóm-thực-hiện)

---

## 📌 1. Giới thiệu tổng quan & Động lực nghiên cứu

Trong các hệ thống Deep Learning quy mô lớn hiện nay (Transformer, LLM, Diffusion Models), chi phí tính toán không chỉ nằm ở các phép toán nhân ma trận (GEMM/MatMul) mà còn bị chi phối nặng nề bởi **chuỗi các toán tử kích hoạt, chuẩn hóa và khử ngẫu nhiên** (Pointwise/Element-wise operators như `BiasAdd`, `ReLU`, `GELU`, `LayerNorm`, `Dropout`).

Trong mô hình **Eager Execution** truyền thống (PyTorch 1.x):
* Mỗi phép toán là một kernel CUDA độc lập.
* Dữ liệu kết quả trung gian giữa các phép tính liên tiếp bắt buộc phải ghi ra bộ nhớ toàn cục (**Global Memory / VRAM**), rồi kernel kế tiếp lại nạp lại từ VRAM.
* Quá trình này gây lãng phí nghiêm trọng băng thông bộ nhớ và tạo độ trễ điều phối kernel (**Kernel Launch Overhead**).

**Dự án này hiện thực một Mini Deep Learning Compiler hoàn chỉnh từ con số 0**, tích hợp:
1. Động cơ tự động tính đạo hàm ngược (**Reverse-mode Automatic Differentiation**) trên đồ thị tính toán có hướng (DAG).
2. Thuật toán tối ưu hóa đồ thị (**Kernel Fusion Pass**) tự động nhận diện và gộp các chuỗi toán tử pointwise liên tiếp thành một siêu node.
3. Bộ sinh mã nguồn **CUDA C++ tối ưu hóa thanh ghi (Registers)** lúc runtime.
4. Trình biên dịch thời gian thực (**NVRTC JIT**) nạp trực tiếp mã máy CUBIN vào GPU NVIDIA mà không cần cài đặt compiler `nvcc` hệ thống.

---

## ⚡ 2. Bản chất bài toán: Memory-Bound & Roofline Model

### 2.1. Cường độ tính toán (Arithmetic Intensity)
Theo mô hình **Roofline**, hiệu năng tối đa có thể đạt được của một kernel bị giới hạn bởi:
$$\text{Performance} = \min\left(\text{Peak Compute FLOPs}, \quad \text{Memory Bandwidth} \times \text{Arithmetic Intensity}\right)$$

* Các toán tử pointwise chỉ thực hiện $1$ phép toán số học trên mỗi phần tử `float32` ($4$ bytes):
  $$\text{Arithmetic Intensity} = \frac{1 \text{ FLOP}}{4 \text{ Bytes}} = \mathbf{0.25 \text{ FLOP/Byte}}$$
* GPU NVIDIA GeForce RTX 3060 có:
  * Năng lực tính toán cực đại FP32: **$12.74 \text{ TFLOPS}$**.
  * Băng thông bộ nhớ VRAM cực đại: **$360 \text{ GB/s}$**.
* Điểm bão hòa tính toán (Ridge Point) của RTX 3060 là:
  $$\text{Ridge Point} = \frac{12,740 \text{ GFLOPS}}{360 \text{ GB/s}} \approx \mathbf{35.4 \text{ FLOP/Byte}}$$

> **Kết luận:** Vì $0.25 \ll 35.4$, toàn bộ các phép toán pointwise nằm sâu trong vùng **Nghẽn cổ chai bộ nhớ (Memory-Bound)**. Các nhân CUDA Cores gần như phải dừng chờ dữ liệu nạp từ VRAM.

### 2.2. So sánh cơ chế Eager Execution vs Kernel Fusion

Xét chuỗi tính toán: $Y = \text{Dropout}(\text{ReLU}(X \cdot W + B))$ với $N$ phần tử:

```
[ Mô hình 1: Eager Execution (4 Kernels rời rạc) ]
  X, W ──► [Kernel 1: Mul]     ──► Ghi tmp1 ra VRAM (1N)
  tmp1, B ──► [Kernel 2: Add]  ──► Ghi tmp2 ra VRAM (1N)
  tmp2 ──► [Kernel 3: ReLU]    ──► Ghi tmp3 ra VRAM (1N)
  tmp3 ──► [Kernel 4: Dropout] ──► Ghi Y ra VRAM (1N)
  => Tổng đọc/ghi VRAM: (2N + 1N) + (2N + 1N) + (1N + 1N) + (1N + 1N) = 10N phần tử = 40N bytes.

[ Mô hình 2: Kernel Fusion (1 Kernel duy nhất) ]
  X, W, B ──► [ Fused Kernel: Mul -> Add -> ReLU -> Dropout ] ──► Ghi Y ra VRAM (1N)
              (tmp1, tmp2, tmp3 nằm hoàn toàn trên THANH GHI)
  => Tổng đọc/ghi VRAM: (3N đọc + 1N ghi) = 4N phần tử = 16N bytes.
```

$$\mathbf{\Delta_{\text{Tiết kiệm VRAM}} = \frac{40N - 16N}{40N} = 60.0\% \text{ lưu lượng bộ nhớ!}}$$

---

## 🏗️ 3. Kiến trúc hệ thống Mini Compiler 4 tầng

```
               [ User Python Code: (x * w + b).relu().dropout() ]
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 1. FRONTEND: COMPUTATIONAL GRAPH & AUTOGRAD (`mini_compiler/core/`)       │
│    - Tensor Class: lưu trữ bộ đệm CuPy/NumPy, `grad`, con trỏ `creator`   │
│    - Các Op nguyên thủy: Add, Mul, ReLU, Dropout, MatMul                  │
│    - DFS Reverse Topological Sort: xây dựng thứ tự lan truyền VJP ngược   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │  Computational Graph (DAG)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 2. MID-END: KERNEL FUSION OPTIMIZER (`mini_compiler/optimizer/`)          │
│    - Operator Classifier: tách nhóm `is_fusible` (Pointwise) và Non-fuse  │
│    - Greedy Cluster Algorithm: gom chuỗi node pointwise liên tiếp         │
│    - External I/O Resolver: loại bỏ toàn bộ tensor trung gian khỏi RAM    │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │  Optimized DAG (FusedOp Subgraphs)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 3. BACKEND: CUDA C++ CODE GENERATOR (`mini_compiler/codegen/`)            │
│    - Ánh xạ I/O ngoài thành con trỏ `__restrict__`                        │
│    - Cấp phát biến thanh ghi cục bộ (`float reg_...`) cho biến trung gian │
│    - Nhúng thuật toán LCG PRNG trực tiếp trên thread cho Dropout          │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │  Mã nguồn C++ CUDA dạng string
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 4. JIT RUNTIME & DRIVER EXECUTION (`mini_compiler/runtime/`)              │
│    - NVRTC JIT Compiler: biên dịch mã C++ sang CUBIN (`sm_86`)            │
│    - CuPy RawKernel / Driver API: `cuModuleLoadData`, `cuLaunchKernel`     │
│    - Compilation Hash Cache: lưu đệm tránh biên dịch lại mã nguồn         │
│    - CUDA Events Benchmark: đo đạc vi giây độ trễ phần cứng GPU          │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 4. Mã nguồn CUDA C++ tự sinh lúc chạy (JIT Generated Code)

Dưới đây là đoạn mã nguồn CUDA C++ thực tế được `CUDACodeGenerator` tự động sản sinh trong RAM lúc chạy chương trình cho chuỗi $X \cdot W + B \to \text{ReLU} \to \text{Dropout}$:

```cuda
extern "C" __global__
void kernel_fused_mul_add_relu_dropout(
    const float* __restrict__ in_0, // Con trỏ Tensor X (Global Memory)
    const float* __restrict__ in_1, // Con trỏ Tensor W (Global Memory)
    const float* __restrict__ in_2, // Con trỏ Tensor B (Global Memory)
    float* __restrict__ out_0,      // Con trỏ Output Y (Global Memory)
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    // -------------------------------------------------------------
    // GIAI ĐOẠN 1: Đọc dữ liệu đúng 1 LẦN DUY NHẤT từ VRAM vào Thanh ghi
    // -------------------------------------------------------------
    float reg_in_0 = in_0[idx];
    float reg_in_1 = in_1[idx];
    float reg_in_2 = in_2[idx];

    // -------------------------------------------------------------
    // GIAI ĐOẠN 2: Toàn bộ tính toán diễn ra trên Thanh ghi (0 byte VRAM)
    // -------------------------------------------------------------
    float reg_mul = reg_in_0 * reg_in_1;
    float reg_add = reg_mul + reg_in_2;
    float reg_relu = (reg_add > 0.0f) ? reg_add : 0.0f;

    // Thuật toán PRNG LCG sinh số ngẫu nhiên ngay trong thread cho Dropout
    // Không cần cấp phát mask trên CPU, không đọc mask từ VRAM
    unsigned int _seed = 42u + (unsigned int)idx;
    _seed = _seed * 1664525u + 1013904223u;
    float _rand_val = (float)(_seed & 0x00FFFFFF) / 16777216.0f;
    float reg_dropout = (_rand_val >= 0.2f) ? (reg_relu * 1.25f) : 0.0f;

    // -------------------------------------------------------------
    // GIAI ĐOẠN 3: Ghi kết quả cuối cùng đúng 1 LẦN DUY NHẤT ra VRAM
    // -------------------------------------------------------------
    out_0[idx] = reg_dropout;
}
```

---

## 🛠️ 5. Engineering Log: 3 Thách thức kỹ thuật thực tế trên GPU Server

Khi chuyển từ môi trường mô phỏng sang thực thi trên server Linux với GPU NVIDIA RTX 3060 thực tế, nhóm đã gặp và giải quyết 3 bài toán hạ tầng:

| STT | Vấn đề phát sinh | Nguyên nhân gốc rễ | Giải pháp kỹ thuật đã áp dụng |
| :---: | :--- | :--- | :--- |
| **1** | `ImportError: libcurand.so.10` | Lệnh `cp.random.randn()` của CuPy mặc định tìm kiếm thư viện CUDA 10 cũ không tồn tại trên hệ thống CUDA 12. | Chuyển đổi toàn bộ việc khởi tạo bộ nhớ sang `cp.empty` + `fill()`. Đối với Dropout trong kernel CUDA, cài đặt thuật toán **PRNG LCG trực tiếp trên thanh ghi**, loại bỏ hoàn toàn sự phụ thuộc vào thư viện `cuRAND`. |
| **2** | `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` | Gói pip `nvidia-cuda-nvrtc-cu12` mang bản 12.8 sinh ra mã trung gian PTX ISA 8.6, trong khi Driver 550.163 của server hỗ trợ tối đa CUDA 12.4 (PTX ISA 8.4). Driver từ chối JIT ngược từ PTX. | Sử dụng backend `cp.RawKernel` để ép NVRTC biên dịch thẳng ra **mã máy nhị phân CUBIN (`sm_86`)** nhắm thẳng kiến trúc phần cứng của RTX 3060. Driver nạp CUBIN trực tiếp, bỏ qua hoàn toàn xung đột phiên bản PTX. |
| **3** | Độ trễ biên dịch JIT lần đầu ($10-30\text{ ms}$) | Quá trình phân tích cú pháp C++ và sinh mã máy của NVRTC tốn tài nguyên CPU. | Thiết kế cấu trúc lưu đệm **JIT Compilation Cache** bằng bảng băm `(cuda_source, kernel_name)`. Các lần gọi kernel ở các epoch tiếp theo đạt thời gian khởi tạo $\approx 0\text{ ms}$. |

---

## 📊 6. Kết quả đo đạc thực nghiệm trên NVIDIA RTX 3060

### 6.1. Bảng số liệu đo đạc thực tế ($N = 10,000,000$ phần tử `float32`)
Thử nghiệm được thực hiện bằng **CUDA Events** với $15$ vòng warm-up và trung bình $50$ lần đo:

| Chỉ số đo đạc | Non-Fused (4 Kernels riêng rẽ) | Fused Kernel (Mini Compiler NVRTC) | Mức cải thiện / Tiết kiệm |
| :--- | :---: | :---: | :---: |
| **Thời gian thực thi (Latency)** | **$1.2084\text{ ms}$** | **$0.4765\text{ ms}$** | **Tăng tốc $2.54\times$ lần** |
| **Lưu lượng đọc/ghi VRAM** | **$381.5\text{ MB}$** | **$152.6\text{ MB}$** | **Tiết kiệm $60.0\%$ VRAM traffic** |
| **Băng thông thực tế đạt được** | $331.02\text{ GB/s}$ | **$335.77\text{ GB/s}$** | Đạt **$93.27\%$** trần vật lý ($360\text{ GB/s}$) |
| **Số lần gọi Kernel Launch** | $4$ lần qua PCIe | **$1$ lần duy nhất** | Giảm $75\%$ chi phí điều phối |

### 6.2. Phân tích độ khớp giữa Lý thuyết và Thực nghiệm
* **Tỉ lệ giảm lưu lượng VRAM lý thuyết:**  
  $$\frac{381.5\text{ MB}}{152.6\text{ MB}} = \mathbf{2.50\times}$$
* **Tỉ lệ tăng tốc độ thực tế đo bằng CUDA Events:**  
  $$\frac{1.2084\text{ ms}}{0.4765\text{ ms}} = \mathbf{2.54\times}$$

> **Độ khớp đạt $> 98.4\%$**: Khi cả hai kernel đều đã chạm trần băng thông phần cứng ($~335\text{ GB/s}$ so với đỉnh $360\text{ GB/s}$), thời gian thực thi tỷ lệ thuận tuyệt đối với dung lượng dữ liệu cần nạp. Phần tăng thêm từ $2.50\times$ lên $2.54\times$ chính là thời gian tiết kiệm được do **loại bỏ 3 lần gọi `cuLaunchKernel`**.

---

## 🔬 7. Đối chiếu kiến trúc với TorchInductor, Triton và XLA

| Tiêu chí so sánh | Mini Compiler (Đề tài) | PyTorch TorchInductor | OpenAI Triton | Google XLA |
| :--- | :--- | :--- | :--- | :--- |
| **Mô hình lập trình** | Python Tensor API + DAG | PyTorch 2.x `torch.compile` | Python Block-level Kernel | TensorFlow / JAX Primitive |
| **Đồ thị IR trung gian**| Custom Topological DAG | FX Graph | Triton IR (MLIR dialect) | HLO (High-Level Optimizer) |
| **Thuật toán Fusion** | Greedy Pointwise Grouping | Pattern Matching trên FX Graph | Tự động hóa qua Block tiling | HLO Instruction Fusion |
| **Ngôn ngữ sinh ra** | **CUDA C++** tối ưu | **Triton Python / C++** | LLVM IR | LLVM NVPTX |
| **Công cụ biên dịch JIT**| **NVRTC (CUDA Runtime)** | C/C++ GCC & Triton JIT | **LLVM JIT Engine** | **LLVM NVPTX Backend** |
| **Lưu trữ trung gian** | **Thanh ghi (Registers)** | **Registers & Shared Memory**| **Registers & SRAM** | **Thanh ghi (Registers)** |

---

## 📂 8. Cấu trúc thư mục & Hướng dẫn tái lập thực nghiệm

### 8.1. Cấu trúc thư mục repository
```text
ltss/
├── mini_compiler/               # Mã nguồn lõi của trình biên dịch
│   ├── core/                    # Đồ thị tính toán & Động cơ Autograd
│   │   ├── tensor.py            # Tensor data structure, autograd backward, DFS topo sort
│   │   ├── ops.py               # Các toán tử: Add, Mul, ReLU, Dropout, MatMul & C++ expr
│   │   └── graph.py             # DAG tracer, cấu trúc đồ thị & Mermaid visualizer
│   ├── optimizer/               # Mid-end Graph Optimizer
│   │   └── fusion.py            # Kernel Fusion Pass: phân loại & gom cụm siêu node FusedOp
│   ├── codegen/                 # Backend Code Generator
│   │   └── cuda_codegen.py      # Sinh mã CUDA C++, register allocation & LCG PRNG
│   ├── runtime/                 # JIT Runtime Execution
│   │   └── jit_nvrtc.py         # NVRTC compiler, CUBIN loader, CUDA Events timer
│   └── compiler.py              # Pipeline tích hợp toàn diện (MiniCompiler)
│
├── benchmark/                   # Module đo đạc hiệu năng
│   └── benchmark_fusion.py      # Script đo đạc Non-fused vs Fused bằng CUDA Events
│
├── tests/                       # Bộ kiểm thử đơn vị (Unit Tests)
│   ├── test_fusion.py           # Kiểm tra thuật toán gom cụm trên đồ thị DAG
│   ├── test_codegen.py          # Kiểm tra tính đúng đắn của mã nguồn CUDA C++ sinh ra
│   └── test_compiler.py         # Kiểm tra tích hợp đầu-cuối của compiler
│
├── notebooks/                   # Jupyter Notebook chạy trực quan
│   └── mini_compiler_colab.ipynb# Notebook có biểu đồ đo đạc so sánh với PyTorch Eager
│
├── BAO_CAO_KET_QUA.md           # Báo cáo học thuật chi tiết của đồ án
├── ket_qua_benchmark.txt        # File log kết quả đo đạc thực tế từ GPU RTX 3060
└── Mini Compiler cho Deep Learning.pptx # Slide báo cáo thuyết trình (Đã chuẩn hóa 17 slides)
```

### 8.2. Hướng dẫn chạy thử nghiệm

#### Cách 1: Chạy trực tiếp trên GPU Server (Linux / JupyterLab)
```bash
# 1. Cài đặt các gói phụ thuộc
pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 matplotlib pandas

# 2. Thiết lập biến môi trường nạp thư viện NVRTC
export LD_LIBRARY_PATH=$(python -c "import os, nvidia.cuda_nvrtc; print(os.path.dirname(nvidia.cuda_nvrtc.__file__) + '/lib')"):$LD_LIBRARY_PATH

# 3. Chạy benchmark đo đạc thời gian và lưu kết quả
python benchmark/benchmark_fusion.py | tee ket_qua_benchmark.txt
```

#### Cách 2: Chạy kiểm thử trên máy cá nhân không có GPU (CPU Fallback Mode)
Hệ thống tự động kích hoạt cơ chế mô phỏng CPU nếu không phát hiện GPU NVIDIA:
```bash
python tests/test_fusion.py      # Kiểm tra giải thuật gom cụm
python tests/test_codegen.py     # Kiểm tra chuỗi code C++ sinh ra
python tests/test_compiler.py    # Kiểm tra pipeline hoàn chỉnh
```

---

## 👥 9. Thông tin nhóm thực hiện

* **Đề tài:** Xây dựng Mini Compiler cho Deep Learning (Autograd Engine & Kernel Fusion JIT)
* **Nhóm thực hiện:** Nhóm 4
  * **Trần Quang Đạt** - MSSV: 20250344E
  * **Đỗ Chí Thanh** - MSSV: 20250916E
  * **Nguyễn Trọng Tiến** - MSSV: 20224159
  * **Nguyễn Đình Mạnh** - MSSV: 20250912E
* **Giảng viên hướng dẫn:** TS. Phạm Doãn Tĩnh
* **Bộ môn:** Khoa học Máy tính – Trường Công nghệ Thông tin & Truyền thông, Đại học Bách Khoa Hà Nội (HUST).
