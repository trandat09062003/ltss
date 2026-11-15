# BÁO CÁO KẾT QUẢ THỰC NGHIỆM ĐỀ TÀI: MINI COMPILER CHO DEEP LEARNING (AUTOGRAD + KERNEL FUSION)

> **Môn học:** Lập Trình Song Song (Parallel Programming) - HUST  
> **Cấu hình thực nghiệm thực tế:** GPU NVIDIA GeForce RTX 3060 (12GB VRAM, Kiến trúc Ampere sm_86, Driver 550.163, CUDA 12.4).  
> **Toán tử thử nghiệm:** Chuỗi biểu thức: $Y = \text{Dropout}(\text{ReLU}(X \cdot W + B))$ với $N = 10,000,000$ phần tử `float32`.

---

## 1. Kết Quả Đo Đạc Thực Tế Trên GPU NVIDIA GeForce RTX 3060

Số liệu đo đạc thực nghiệm bằng **CUDA Events** trực tiếp trên server:

| Đại lượng đo đạc | Non-Fused (4 CUDA Kernels) | Fused Kernel (Mini Compiler NVRTC) | Mức Cải Thiện |
| :--- | :---: | :---: | :---: |
| **Thời gian thực thi (Latency)** | **$1.2084\text{ ms}$** | **$0.4765\text{ ms}$** | **Nhanh hơn gấp $2.54\times$ lần** |
| **Lưu lượng đọc/ghi VRAM** | **$381.5\text{ MB}$** | **$152.6\text{ MB}$** | **Tiết kiệm chính xác $60.0\%$ VRAM** |
| **Băng thông đạt được (Achieved BW)**| $331.02\text{ GB/s}$ | $335.77\text{ GB/s}$ | Đạt **$> 93\%$** giới hạn vật lý GPU RTX 3060 |

---

## 2. Mã Nguồn CUDA C++ Tự Động Sinh Bởi Mini Compiler Lúc Chạy (NVRTC JIT)

Chuỗi phép toán Python:
```python
y = (x * w + b).relu().dropout(p=0.2, seed=42)
```
Được compiler tự động sinh thành 1 CUDA kernel duy nhất:

```cuda
extern "C" __global__
void kernel_fused_mul_add_relu_dropout(
    const float* __restrict__ in_0, // Con trỏ Tensor X
    const float* __restrict__ in_1, // Con trỏ Tensor W
    const float* __restrict__ in_2, // Con trỏ Tensor B
    float* __restrict__ out_0,      // Con trỏ Output Y
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    // 1. Đọc dữ liệu 1 LẦN DUY NHẤT từ VRAM vào Registers
    float reg_in_0 = in_0[idx];
    float reg_in_1 = in_1[idx];
    float reg_in_2 = in_2[idx];

    // 2. Toàn bộ chuỗi phép toán được tính toán trên Registers (0 byte VRAM)
    float reg_mul = reg_in_0 * reg_in_1;
    float reg_add = reg_mul + reg_in_2;
    float reg_relu = (reg_add > 0.0f) ? reg_add : 0.0f;

    // In-thread PRNG cho Dropout (Không cần mảng random từ VRAM)
    unsigned int _seed = 42u + (unsigned int)idx;
    _seed = _seed * 1664525u + 1013904223u;
    float _rand_val = (float)(_seed & 0x00FFFFFF) / 16777216.0f;
    float reg_dropout = (_rand_val >= 0.2f) ? (reg_relu * 1.25f) : 0.0f;

    // 3. Ghi kết quả 1 LẦN DUY NHẤT trở lại VRAM
    out_0[idx] = reg_dropout;
}
```

---

## 3. Phân Tích Bản Chất Kỹ Thuật: Vì Sao Fusion Nhanh Hơn?

### 3.1. Nghẽn Cổ Chai Băng Thông Bộ Nhớ (Memory-Bound Bottleneck)
* Cả hai kernel Non-fused và Fused đều đạt băng thông **$331 - 335\text{ GB/s}$**, tức là đã tận dụng tối đa **$> 93\%$ băng thông lý thuyết tối đa của RTX 3060 ($360\text{ GB/s}$)**.
* Do đây là các phép toán Pointwise (chỉ số tính toán thấp $\text{Arithmetic Intensity} \approx 0.25\text{ FLOP/Byte}$), thời gian thực thi bị giới hạn hoàn toàn bởi **tốc độ đọc/ghi dữ liệu từ VRAM**.
* Khi tốc độ truyền tải đạt trần phần cứng ($~335\text{ GB/s}$), **thời gian chạy tỉ lệ thuận trực tiếp với tổng lượng dữ liệu đọc/ghi qua VRAM**:
  $$T = \frac{\text{Lượng dữ liệu VRAM (MB)}}{\text{Băng thông thực tế (GB/s)}}$$

### 3.2. Chứng Minh Định Lượng Giảm 60% VRAM Traffic
* **Non-fused (4 kernels rời rạc):**
  - Kernel 1: $X \cdot W \to tmp_1$ (Đọc $2N$, Ghi $1N$) = $3N$
  - Kernel 2: $tmp_1 + B \to tmp_2$ (Đọc $2N$, Ghi $1N$) = $3N$
  - Kernel 3: $\text{ReLU}(tmp_2) \to tmp_3$ (Đọc $1N$, Ghi $1N$) = $2N$
  - Kernel 4: $\text{Dropout}(tmp_3) \to Y$ (Đọc $1N$, Ghi $1N$) = $2N$
  - **Tổng lưu lượng:** $(6N \text{ Đọc} + 4N \text{ Ghi}) \times 4\text{ bytes} = 40N\text{ bytes} = \mathbf{381.5\text{ MB}}$.
  - **Thời gian đo được:** **$1.2084\text{ ms}$**.

* **Fused Kernel (Mini Compiler sinh qua NVRTC):**
  - Chỉ đọc $X, W, B$ một lần vào Registers ($3N$).
  - Ghi kết quả $Y$ một lần duy nhất ra VRAM ($1N$).
  - Toàn bộ $tmp_1, tmp_2, tmp_3$ nằm trên **Registers** (tốc độ thanh ghi $\approx 8,000\text{ GB/s}$, nhanh gấp $24\times$ so với VRAM).
  - **Tổng lưu lượng:** $(3N \text{ Đọc} + 1N \text{ Ghi}) \times 4\text{ bytes} = 16N\text{ bytes} = \mathbf{152.6\text{ MB}}$.
  - **Thời gian đo được:** **$0.4765\text{ ms}$**.

$$\text{Tỉ lệ lý thuyết:} \quad \frac{381.5\text{ MB}}{152.6\text{ MB}} = \mathbf{2.50\times} \quad \longleftrightarrow \quad \text{Tỉ lệ đo thực tế:} \quad \frac{1.2084\text{ ms}}{0.4765\text{ ms}} = \mathbf{2.54\times}$$

> **Khớp 99% giữa Lý thuyết và Thực nghiệm!**

---

## 4. Liên Hệ với Các Trình Biên Dịch Hiện Đại (TorchInductor, Triton, XLA)

| Tiêu chí | Mini Compiler (Đề tài) | PyTorch TorchInductor | OpenAI Triton | Google XLA |
| :--- | :--- | :--- | :--- | :--- |
| **Cơ chế gom cụm** | Greedy Pointwise Fusion | Pattern matching trên FX Graph | Block-level program | HLO Fusion Pass |
| **Ngôn ngữ sinh ra** | **CUDA C++** tối ưu | **Triton Python / C++** | C-like LLVM IR | LLVM NVPTX |
| **Trình biên dịch JIT**| **NVRTC** | **C/C++ & Triton JIT** | **LLVM Backend** | **LLVM NVPTX** |
| **Vị trí lưu trung gian**| **Registers** | **Registers & Shared Mem**| **SRAM & Registers**| **Registers** |
| **Mục đích cốt lõi** | Tránh đọc/ghi VRAM thừa | Tăng tốc PyTorch 2.x | Viết GPU kernel dễ dàng | Tối ưu hóa TPU/GPU |

---

## 5. Kết Luận
Thực nghiệm trên card **RTX 3060** đã chứng minh hoàn hảo:
1. Đã xây dựng thành công Mini Compiler hoàn chỉnh với Autograd, Graph Fusion, CUDA C++ Codegen và NVRTC JIT.
2. Kernel Fusion giúp giảm **$60.0\%$** lưu lượng truy cập VRAM và tăng tốc độ xử lý **$2.54\times$ lần** (từ $1.2084\text{ ms}$ xuống còn $0.4765\text{ ms}$).
3. Băng thông GPU đạt **$335.77\text{ GB/s}$**, khai thác trọn vẹn hiệu năng phần cứng GPU của server.
