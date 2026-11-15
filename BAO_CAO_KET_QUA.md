# BÁO CÁO BÀI TẬP LỚN: LẬP TRÌNH SONG SONG
## ĐỀ TÀI: XÂY DỰNG MINI COMPILER CHO DEEP LEARNING (AUTOGRAD ENGINE VÀ KERNEL FUSION JIT)

* **Học phần:** Lập trình song song (Parallel Programming)
* **Khoa / Trường:** Trường Công nghệ Thông tin & Truyền thông – Đại học Bách Khoa Hà Nội (HUST)
* **Môi trường thực nghiệm:** Server Linux, GPU NVIDIA GeForce RTX 3060 (12GB GDDR6, 192-bit bus, Driver 550.163.01, CUDA 12.4)

---

## MỤC LỤC
1. [Đặt vấn đề và Động lực nghiên cứu](#1-đặt-vấn-đề-và-động-lực-nghiên-cứu)
2. [Kiến trúc tổng thể của Mini Compiler](#2-kiến-trúc-tổng-thể-của-mini-compiler)
3. [Thiết kế & Hiện thực kỹ thuật](#3-thiết-kế--hiện-thực-kỹ-thuật)
   - 3.1. Đồ thị tính toán DAG và Động cơ Autograd (Reverse-mode AD)
   - 3.2. Thuật toán gom cụm toán tử (Kernel Fusion Pass)
   - 3.3. Bộ sinh mã nguồn CUDA C++ (Codegen Engine)
   - 3.4. JIT Compilation và Nạp mã thực thi qua NVRTC Driver API
4. [Các thách thức kỹ thuật phát sinh và giải pháp](#4-các-thách-thức-kỹ-thuật-phát-sinh-và-giải-pháp)
5. [Đánh giá thực nghiệm và Phân tích hiệu năng](#5-đánh-giá-thực-nghiệm-và-phân-tích-hiệu-năng)
   - 5.1. Thiết lập kịch bản đo đạc
   - 5.2. Kết quả đo đạc thực tế trên GPU RTX 3060
   - 5.3. Phân tích định lượng băng thông bộ nhớ (Memory Bandwidth & Roofline)
6. [Liên hệ với các hệ thống compiler hiện đại (TorchInductor, Triton, XLA)](#6-liên-hệ-với-các-hệ-thống-compiler-hiện-đại)
7. [Kết luận và Hướng mở rộng](#7-kết-luận-và-hướng-mở-rộng)

---

## 1. Đặt vấn đề và Động lực nghiên cứu

Trong kiến trúc tính toán của các mô hình Deep Learning hiện đại (đặc biệt là Transformer, CNN hay MLP), đồ thị tính toán thường chứa đan xen hai nhóm toán tử chính:
1. **Toán tử tính toán nặng (Compute-bound):** Điển hình là phép nhân ma trận (GEMM / MatMul) và tích chập (Convolution). Các toán tử này có cường độ tính toán (Arithmetic Intensity) cao, phần lớn thời gian thực thi nằm ở các đơn vị xử lý Tensor Cores / CUDA Cores.
2. **Toán tử xử lý theo từng phần tử (Memory-bound / Pointwise):** Ví dụ như Bias Addition ($+ b$), hàm kích hoạt ($\text{ReLU}, \text{GELU}$), chuẩn hóa ($\text{LayerNorm}$) và khử ngẫu nhiên ($\text{Dropout}$). Các toán tử này chỉ thực hiện 1 hoặc 2 phép tính số học trên mỗi phần tử 4-byte (`float32`), khiến Arithmetic Intensity cực kỳ thấp:
   $$\text{Arithmetic Intensity} = \frac{\text{Số phép toán (FLOPs)}}{\text{Dung lượng đọc/ghi (Bytes)}} \approx \frac{1 \text{ FLOP}}{4 \text{ Bytes}} = 0.25 \text{ FLOP/Byte}$$

Trong các framework dạng **Eager Execution** (như PyTorch bản tiêu chuẩn trước 2.0):
* Mỗi toán tử được biên dịch thành một CUDA kernel riêng biệt.
* Giữa hai toán tử liên tiếp (ví dụ: $tmp_1 = X \cdot W + B$ rồi $tmp_2 = \text{ReLU}(tmp_1)$), kernel thứ nhất bắt buộc phải đẩy toàn bộ dữ liệu kết quả trung gian ra bộ nhớ toàn cục (Global Memory/VRAM) của GPU. Kernel kế tiếp sau đó lại nạp chính mảng trung gian đó từ VRAM trở lại chip.
* Quá trình đọc/ghi mảng trung gian này gây ra hiện tượng **nghẽn cổ chai băng thông bộ nhớ (Memory Bandwidth Bottleneck)**, đồng thời tạo ra chi phí điều phối (kernel launch overhead) qua giao tiếp PCIe/Driver.

**Giải pháp Kernel Fusion:**  
Gộp toàn bộ chuỗi toán tử pointwise liên tiếp (ví dụ: $X \cdot W + B \to \text{ReLU} \to \text{Dropout}$) thành **một CUDA kernel duy nhất**. Dữ liệu đầu vào chỉ cần đọc từ VRAM đúng một lần vào các thanh ghi (Registers) của thread, toàn bộ chuỗi phép tính được thực hiện trên thanh ghi với độ trễ cực thấp, và chỉ ghi kết quả cuối cùng một lần ra VRAM.

Đề tài này tập trung xây dựng một hệ thống **Mini Deep Learning Compiler** khép kín từ đầu đến cuối, bao gồm đồ thị tính toán (DAG), động cơ đạo hàm tự động (Autograd), thuật toán tối ưu gom cụm (Kernel Fusion Pass), bộ sinh mã CUDA C++ lúc runtime và biên dịch JIT thông qua thư viện NVRTC của NVIDIA.

---

## 2. Kiến trúc tổng thể của Mini Compiler

Hệ thống được thiết kế theo mô hình compiler 4 giai đoạn chuẩn mực:

```
[ Frontend: Python API ]
   │  (Định nghĩa Tensor, forward pass, reverse-mode autodiff)
   ▼
[ Intermediate Representation (IR): Computational Graph (DAG) ]
   │  (Các Node toán tử Add, Mul, ReLU, Dropout, MatMul và Edges truyền tensor)
   ▼
[ Mid-end: Graph Optimizer (Kernel Fusion Pass) ]
   │  (Phân loại toán tử, nhận diện chuỗi fusible, gộp thành FusedPointwiseOp)
   ▼
[ Backend: CUDA Code Generator & NVRTC JIT Runtime ]
   │  (Sinh mã nguồn CUDA C++, cấp phát thanh ghi, biên dịch sang PTX/Machine code)
   ▼
[ Hardware Execution: NVIDIA GPU Driver API ]
   │  (Cấp phát VRAM, thiết lập Grid/Block dimension, thực thi kernel và benchmark)
```

Cấu trúc module trong mã nguồn:
* `mini_compiler/core/`: Định nghĩa lớp `Tensor`, các toán tử nguyên thủy `Op`, và lớp quản lý đồ thị `ComputationalGraph`.
* `mini_compiler/optimizer/`: Chứa lớp `FusionOptimizer` thực hiện pass quét đồ thị và gom cụm thành `FusedOp`.
* `mini_compiler/codegen/`: Chứa `CUDACodeGenerator` chịu trách nhiệm chuyển đổi cấu trúc `FusedOp` thành mã nguồn CUDA C++.
* `mini_compiler/runtime/`: Module `jit_nvrtc.py` bọc thư viện NVRTC, nạp kernel vào GPU context và đo đạc thời gian bằng CUDA Events.
* `mini_compiler/compiler.py`: Điểm kết nối toàn bộ quy trình thành một pipeline hoàn chỉnh.

---

## 3. Thiết kế & Hiện thực kỹ thuật

### 3.1. Đồ thị tính toán DAG và Động cơ Autograd (Reverse-mode AD)
Mỗi đối tượng `Tensor` đại diện cho một cạnh (edge) mang dữ liệu hoặc gradient trong đồ thị. Nếu một tensor là kết quả của một phép toán, nó lưu con trỏ `creator` trỏ đến node `Op` tương ứng.

* **Topological Sort:** Để thực hiện đạo hàm ngược (backward pass), hệ thống thực hiện duyệt đồ thị theo chiều sâu (DFS) bắt đầu từ node Loss để thiết lập danh sách sắp xếp tô-pô.
* **Quy tắc dây chuyền (Vector-Jacobian Product - VJP):**  
  Mỗi toán tử cài đặt phương thức `backward(*out_grads)` tương ứng:
  * $\text{Add}(A, B)$: $g_A = g_{out}, \quad g_B = g_{out}$ (kèm logic xử lý broadcast nếu lệch rank).
  * $\text{Mul}(A, B)$: $g_A = g_{out} \cdot B, \quad g_B = g_{out} \cdot A$.
  * $\text{ReLU}(A)$: $g_A = g_{out} \cdot \mathbb{I}(A > 0)$.
  * $\text{Dropout}(A, p)$: $g_A = g_{out} \cdot \frac{\text{mask}}{1 - p}$.
  * $\text{MatMul}(A, B)$: $g_A = g_{out} \cdot B^T, \quad g_B = A^T \cdot g_{out}$.

### 3.2. Thuật toán gom cụm toán tử (Kernel Fusion Pass)
Bộ tối ưu hóa `FusionOptimizer` duyệt danh sách các node theo thứ tự tô-pô và áp dụng chiến lược gom cụm Greedy:

1. **Phân loại toán tử:**
   * Thuộc tính `is_fusible = True`: Gồm các toán tử pointwise cùng chiều hoặc broadcast được (`Add`, `Mul`, `ReLU`, `Dropout`).
   * Thuộc tính `is_fusible = False`: Các toán tử đòi hỏi đồng bộ warp/block hoặc tính toán phức tạp như `MatMul`.
2. **Xác định biên cụm (Cluster Boundary):**
   * Duyệt qua từng node: Nếu node hiện tại có `is_fusible == True`, đưa vào cụm tạm thời `current_cluster`.
   * Khi gặp node `is_fusible == False` hoặc hết đồ thị: Thực hiện chốt cụm (`flush_cluster`).
3. **Phân giải I/O ngoài cụm (External Inputs/Outputs Resolution):**
   * *External Inputs:* Mọi tensor đầu vào của các sub-node trong cụm mà không do bất kỳ sub-node nào khác trong cụm đó sinh ra.
   * *External Outputs:* Các tensor kết quả của sub-node mà được tiêu thụ bởi các node bên ngoài cụm hoặc là tensor đầu ra cuối cùng của đồ thị.
   * Toàn bộ các tensor trung gian khác (chỉ dùng nội bộ trong cụm) bị loại bỏ hoàn toàn khỏi I/O của kernel. Chúng sẽ được cấp phát thành các biến cục bộ (registers) trong CUDA kernel.

### 3.3. Bộ sinh mã nguồn CUDA C++ (Codegen Engine)
Lớp `CUDACodeGenerator` chuyển đổi một `FusedOp` thành một chuỗi mã nguồn C++ hoàn chỉnh với các đặc tính tối ưu:
* **`__restrict__` qualifiers:** Khai báo trên tất cả các con trỏ đầu vào/đầu ra để báo cho trình biên dịch NVIDIA rằng các vùng nhớ không bị chồng lấn (aliasing), cho phép sinh chỉ thị nạp dữ liệu vector (ví dụ `LDG.E.128`).
* **Cấp phát biến thanh ghi:** Ánh xạ các tensor trung gian thành biến cục bộ kiểu `float`.
* **Cơ chế PRNG nhúng trực tiếp trong thread cho Dropout:** Thay vì sinh mask trên CPU hoặc đọc từ VRAM, mỗi thread tính toán một số giả ngẫu nhiên bằng thuật toán Linear Congruential Generator (LCG) dựa trên Thread ID và Seed:
  ```cuda
  unsigned int _seed = 42u + (unsigned int)idx;
  _seed = _seed * 1664525u + 1013904223u;
  float _rand_val = (float)(_seed & 0x00FFFFFF) / 16777216.0f;
  float reg_dropout = (_rand_val >= 0.2f) ? (reg_relu * 1.25f) : 0.0f;
  ```

Đoạn mã CUDA C++ tự sinh hoàn chỉnh cho chuỗi $X \cdot W + B \to \text{ReLU} \to \text{Dropout}$:
```cuda
extern "C" __global__
void kernel_fused_mul_add_relu_dropout(
    const float* __restrict__ in_0, // X
    const float* __restrict__ in_1, // W
    const float* __restrict__ in_2, // B
    float* __restrict__ out_0,      // Output Y
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    // 1. Đọc dữ liệu đúng 1 lần từ VRAM vào thanh ghi (Registers)
    float reg_in_0 = in_0[idx];
    float reg_in_1 = in_1[idx];
    float reg_in_2 = in_2[idx];

    // 2. Toàn bộ chuỗi phép toán được tính toán trên Registers (0 byte VRAM)
    float reg_mul = reg_in_0 * reg_in_1;
    float reg_add = reg_mul + reg_in_2;
    float reg_relu = (reg_add > 0.0f) ? reg_add : 0.0f;

    unsigned int _seed = 42u + (unsigned int)idx;
    _seed = _seed * 1664525u + 1013904223u;
    float _rand_val = (float)(_seed & 0x00FFFFFF) / 16777216.0f;
    float reg_dropout = (_rand_val >= 0.2f) ? (reg_relu * 1.25f) : 0.0f;

    // 3. Ghi kết quả đúng 1 lần ra VRAM
    out_0[idx] = reg_dropout;
}
```

### 3.4. JIT Compilation và Nạp mã thực thi qua NVRTC Driver API
Quá trình biên dịch mã nguồn diễn ra trực tiếp trong RAM lúc chạy chương trình:
1. Chuỗi mã nguồn CUDA C++ được nạp vào NVRTC program bằng `nvrtcCreateProgram`.
2. Trình biên dịch NVRTC chạy với các cờ tối ưu hóa: `--use_fast_math`, `-std=c++14`, và kiến trúc tương ứng (`sm_86` cho RTX 3060).
3. Kết quả biên dịch (CUBIN / binary machine code) được nạp trực tiếp vào GPU Context thông qua CUDA Driver API (`cuModuleLoadData`).
4. Khi chạy, kernel function handle được gọi với cấu hình khối tiêu chuẩn: `blockDim = (256, 1, 1)` và `gridDim = (ceil(N / 256), 1, 1)`.

---

## 4. Các thách thức kỹ thuật phát sinh và giải pháp

Trong quá trình triển khai trên môi trường server thực tế, chúng tôi đã gặp và giải quyết 3 vấn đề kỹ thuật lớn:

1. **Lỗi phụ thuộc thư viện ngẫu nhiên `libcurand.so`:**  
   * *Hiện tượng:* Ban đầu khi khởi tạo dữ liệu ngẫu nhiên thông qua `cp.random.randn()`, hệ thống báo lỗi thiếu thư viện động `libcurand.so.10`.
   * *Nguyên nhân:* Môi trường server chạy CUDA 12, trong khi các lệnh sinh số ngẫu nhiên cấp cao cố gắng tìm liên kết thư viện CUDA 10 cũ.
   * *Giải pháp:* Chúng tôi tái cấu trúc toàn bộ việc khởi tạo bộ nhớ sang sử dụng `cp.empty` kết hợp `fill` hoặc sinh ngẫu nhiên từ phía CPU rồi nạp lên GPU. Đối với phép toán Dropout trong kernel CUDA, giải pháp PRNG LCG trực tiếp trên thanh ghi đã giải quyết hoàn toàn sự phụ thuộc vào cuRAND.

2. **Lỗi bất tương thích phiên bản PTX (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`):**  
   * *Hiện tượng:* NVRTC sinh ra mã trung gian PTX, nhưng khi gọi `cuModuleLoadData` thì CUDA Driver từ chối nạp mã với mã lỗi trên.
   * *Nguyên nhân:* Gói `nvidia-cuda-nvrtc-cu12` cài đặt qua pip mang phiên bản 12.8 (sinh mã PTX ISA 8.6), trong khi NVIDIA Display Driver của server là bản 550.163 (hỗ trợ tối đa CUDA 12.4, tương ứng PTX ISA 8.4).
   * *Giải pháp:* Thay vì để NVRTC sinh mã trung gian PTX để driver JIT ngược lại, hệ sinh thái CuPy RawKernel được cấu hình để biên dịch trực tiếp sang **mã máy CUBIN (`sm_86`)**. Mã nhị phân nhắm đúng kiến trúc phần cứng của RTX 3060 nên driver nạp và thực thi trực tiếp, vượt qua hoàn toàn xung đột phiên bản PTX.

3. **Cơ chế lưu đệm biên dịch (JIT Cache):**  
   * Quá trình biên dịch qua NVRTC tốn khoảng $10 - 30\text{ ms}$ cho lần đầu tiên. Hệ thống đã bổ sung một bảng băm `_compiled_cache` dựa trên `(cuda_source, kernel_name)` để tái sử dụng function handle ở các bước lặp huấn luyện kế tiếp mà không phải biên dịch lại.

---

## 5. Đánh giá thực nghiệm và Phân tích hiệu năng

### 5.1. Thiết lập kịch bản đo đạc
* **Thiết bị:** NVIDIA GeForce RTX 3060 (12GB VRAM, 3584 CUDA Cores, Peak Memory Bandwidth lý thuyết: $360\text{ GB/s}$).
* **Kích thước dữ liệu kiểm thử:** $N = 10,000,000$ phần tử kiểu `float32` ($40\text{ MB}$ cho mỗi tensor $1D$).
* **Quy trình đo đạc:**
  * Thực hiện **20 vòng chạy khởi động (Warm-up)** để nạp dữ liệu lên GPU cache và ổn định xung nhịp GPU.
  * Sử dụng **CUDA Events** (`cudaEventRecord`, `cudaEventElapsedTime`) với độ phân giải microsecond để đo thời gian chạy thực tế trên chip GPU, đồng bộ hóa bằng `cudaStreamSynchronize`. Không dùng `time.time()` của Python để tránh sai số do cơ chế bất đồng bộ của CUDA.
  * Lặp lại **100 lần** và lấy giá trị trung bình.

### 5.2. Kết quả đo đạc thực tế trên GPU RTX 3060

| Phương pháp thực thi | Thời gian thực thi (Latency) | Lưu lượng VRAM truy cập | Băng thông đạt được (Achieved BW) | Tốc độ tăng (Speedup) |
| :--- | :---: | :---: | :---: | :---: |
| **Non-Fused (4 Kernels riêng biệt)** | **$1.2084\text{ ms}$** | **$381.5\text{ MB}$** | $331.02\text{ GB/s}$ | $1.00\times$ (Cơ sở) |
| **Fused Kernel (Mini Compiler NVRTC)** | **$0.4765\text{ ms}$** | **$152.6\text{ MB}$** | **$335.77\text{ GB/s}$** | **$2.54\times$ (Nhanh hơn 254%)** |

### 5.3. Phân tích định lượng băng thông bộ nhớ (Memory Bandwidth & Roofline)

Kết quả thực nghiệm cho thấy mức tăng tốc độ **$2.54\times$** khớp gần như tuyệt đối với dự đoán lý thuyết ($2.50\times$). Dưới đây là phân tích chi tiết:

1. **Phân tích số lượt truy cập VRAM:**  
   Với tensor có $N = 10^7$ phần tử `float32` (mỗi phần tử $4$ bytes):
   * **Trong mô hình Non-fused (4 kernels):**
     * Kernel 1 ($\text{Mul}$): Đọc $X, W$ ($2N$), ghi $tmp_1$ ($1N$) $\to 3N$ phần tử.
     * Kernel 2 ($\text{Add}$): Đọc $tmp_1, B$ ($2N$), ghi $tmp_2$ ($1N$) $\to 3N$ phần tử.
     * Kernel 3 ($\text{ReLU}$): Đọc $tmp_2$ ($1N$), ghi $tmp_3$ ($1N$) $\to 2N$ phần tử.
     * Kernel 4 ($\text{Dropout}$): Đọc $tmp_3$ ($1N$), ghi $Y$ ($1N$) $\to 2N$ phần tử.
     * **Tổng lượng dữ liệu đọc/ghi qua VRAM:**  
       $$\text{Traffic}_{\text{Non-fused}} = (3N + 3N + 2N + 2N) \times 4\text{ bytes} = 40N\text{ bytes} = \mathbf{381.47\text{ MB}}$$
   * **Trong mô hình Fused (1 kernel duy nhất):**
     * Chỉ đọc các đầu vào ban đầu $X, W, B$ một lần duy nhất vào thanh ghi: $3N$ phần tử.
     * Toàn bộ giá trị $tmp_1, tmp_2, tmp_3$ được giữ trên thanh ghi (truy xuất nội bộ trong chip với băng thông hàng nghìn GB/s, không tiêu tốn bus VRAM).
     * Chỉ ghi mảng kết quả cuối cùng $Y$ một lần ra VRAM: $1N$ phần tử.
     * **Tổng lượng dữ liệu đọc/ghi qua VRAM:**  
       $$\text{Traffic}_{\text{Fused}} = (3N + 1N) \times 4\text{ bytes} = 16N\text{ bytes} = \mathbf{152.59\text{ MB}}$$

   $$\text{Mức tiết kiệm lưu lượng VRAM:} \quad \frac{381.47 - 152.59}{381.47} = \mathbf{60.0\%}$$

2. **Khai thác giới hạn phần cứng GPU (Saturating Memory Bandwidth):**  
   Băng thông bộ nhớ thực tế đạt được tính theo công thức:
   $$\text{Bandwidth}_{\text{achieved}} = \frac{\text{Lưu lượng dữ liệu (Bytes)}}{\text{Thời gian thực thi (s)} \times 10^9}$$
   * Non-fused: $\frac{381.47 \times 10^6 \text{ Bytes}}{1.2084 \times 10^{-3} \text{ s}} \approx \mathbf{331.02\text{ GB/s}}$
   * Fused: $\frac{152.59 \times 10^6 \text{ Bytes}}{0.4765 \times 10^{-3} \text{ s}} \approx \mathbf{335.77\text{ GB/s}}$

   Cả hai kịch bản đều đạt xấp xỉ **$92 - 93\%$ băng thông cực đại lý thuyết của RTX 3060 ($360\text{ GB/s}$)**. Điều này khẳng định rằng GPU đang chạy ở ngưỡng trần phần cứng của bài toán Memory-bound. Do tốc độ nạp dữ liệu trên giây là cố định, **khi tổng lượng dữ liệu cần nạp giảm đi $2.5\times$ (từ 381.5 MB xuống 152.6 MB), thời gian thực thi bắt buộc phải giảm tương ứng $2.5\times$**.

3. **Loại bỏ chi phí điều phối (Kernel Launch Overhead):**  
   Mức chênh lệch nhẹ giữa tốc độ tăng đo được ($2.54\times$) so với lý thuyết lưu lượng ($2.50\times$) là nhờ Fused Kernel đã loại bỏ được chi phí của 3 lần gọi `cuLaunchKernel` và chuyển đổi ngữ cảnh liên quan qua bus PCIe.

---

## 6. Liên hệ với các hệ thống compiler hiện đại

Mô hình Mini Compiler được xây dựng trong đề tài phản ánh chính xác các nguyên lý nền tảng của các deep learning compiler tiên tiến nhất hiện nay:

* **TorchInductor (PyTorch 2.x):**  
  TorchInductor sử dụng TorchDynamo để bắt đồ thị FX Graph, sau đó duyệt đồ thị để tìm các cụm phép toán pointwise và sinh mã Triton hoặc C++ OpenMP. Nguyên lý gộp các phép toán liên tiếp vào registers của TorchInductor giống hệt với pass `FusionOptimizer` trong đề tài này.
* **OpenAI Triton:**  
  Triton cho phép lập trình viên viết các khối kernel dạng block-level trong Python rồi biên dịch ra PTX/CUBIN qua LLVM. Kỹ thuật sinh mã C++ có cấu trúc template của chúng tôi tương tự như tầng hạ tầng sinh mã của Triton Compiler.
* **Google XLA (Accelerated Linear Algebra):**  
  XLA thực hiện các biến đổi HLO (High-Level Optimizer) để gom các chuỗi phép toán kích hoạt, chuẩn hóa và phép cộng vào chung một phép toán Fusion.

---

## 7. Kết luận và Hướng mở rộng

### 7.1. Kết luận
1. Đã xây dựng hoàn chỉnh một Mini Compiler cho Deep Learning bao gồm: Engine Autograd (Reverse-mode AD), Biểu diễn đồ thị DAG, Thuật toán Kernel Fusion Pass, Bộ sinh mã CUDA C++ tự động và JIT Runtime qua NVRTC.
2. Đã kiểm chứng thực nghiệm thành công trên card đồ họa **NVIDIA GeForce RTX 3060**:
   * Giảm chính xác **$60.0\%$** lưu lượng đọc/ghi bộ nhớ VRAM.
   * Tăng tốc độ thực thi lên **$2.54\times$ lần** (từ $1.2084\text{ ms}$ xuống $0.4765\text{ ms}$).
   * Khai thác **$> 93\%$** băng thông vật lý của phần cứng ($335.77\text{ GB/s}$).

### 7.2. Hướng mở rộng
* **GEMM-Epilogue Fusion:** Tích hợp chuỗi pointwise activation vào trực tiếp giai đoạn lưu kết quả (Epilogue store) của ma trận tích trong Tiled GEMM hoặc CuBLAS.
* **Reduction Fusion:** Mở rộng pass gom cụm cho các phép toán giảm chiều (Reduction) như `Softmax` và `LayerNorm` thông qua cơ chế Shared Memory và Warp Shuffle (`__shfl_down_sync`).
