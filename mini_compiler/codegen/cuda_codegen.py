import re

class CUDACodeGenerator:
    """
    Bộ sinh mã nguồn CUDA C++ từ FusedOp hoặc Single Op.
    Tối ưu hóa:
      - Tận dụng Registers cho biến trung gian.
      - Sử dụng __restrict__ để hỗ trợ compiler tối ưu hóa vector load/store.
      - Tích hợp fast PRNG trực tiếp trên thread cho Dropout.
    """
    def __init__(self):
        pass

    def generate_fused_kernel(self, fused_op, kernel_name=None):
        """
        Sinh toàn bộ mã nguồn CUDA C++ cho một FusedOp.
        """
        kernel_name = kernel_name or f"kernel_{fused_op.name}"
        kernel_name = re.sub(r'[^a-zA-Z0-9_]', '_', kernel_name)

        ext_inputs = fused_op.inputs
        ext_outputs = fused_op.outputs
        sub_nodes = fused_op.sub_nodes

        # 1. Danh sách tham số hàm CUDA
        param_list = []
        for i, inp in enumerate(ext_inputs):
            param_list.append(f"const float* __restrict__ in_{i}")
        for i, out in enumerate(ext_outputs):
            param_list.append(f"float* __restrict__ out_{i}")
        param_list.append("int N")
        param_signature = ", ".join(param_list)

        # 2. Đọc inputs từ Global Memory vào Registers
        tensor_to_var = {}
        load_code_lines = []
        for i, inp in enumerate(ext_inputs):
            var_name = f"reg_in_{i}"
            tensor_to_var[inp] = var_name
            load_code_lines.append(f"    float {var_name} = in_{i}[idx];")

        # 3. Chuỗi tính toán trên Registers (không ghi ra VRAM)
        compute_code_lines = []
        for op in sub_nodes:
            # Map input tensors sang biến local
            in_vars = [tensor_to_var[inp] for inp in op.inputs]
            out_var = f"reg_{op.op_type}_{id(op) % 1000}"
            # Gán mapping cho output tensor của op này
            for out_tensor in op.outputs:
                tensor_to_var[out_tensor] = out_var

            # Lấy biểu thức CUDA
            expr = op.get_cuda_expr(in_vars, out_var)
            compute_code_lines.append(f"    {expr}")

        # 4. Ghi kết quả ra Global Memory (chỉ ghi các output được yêu cầu)
        store_code_lines = []
        for i, out in enumerate(ext_outputs):
            final_var = tensor_to_var[out]
            store_code_lines.append(f"    out_{i}[idx] = {final_var};")

        # 5. Ráp thành source code CUDA C++ hoàn chỉnh
        loads_str = "\n".join(load_code_lines)
        compute_str = "\n".join(compute_code_lines)
        stores_str = "\n".join(store_code_lines)

        cuda_source = f"""extern "C" __global__
void {kernel_name}({param_signature}) {{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    // 1. Đọc dữ liệu 1 LẦN DUY NHẤT từ Global Memory vào Registers
{loads_str}

    // 2. Toàn bộ chuỗi phép toán được tính toán trên Registers
{compute_str}

    // 3. Ghi kết quả 1 LẦN DUY NHẤT trở lại Global Memory
{stores_str}
}}
"""
        return kernel_name, cuda_source

    def generate_single_op_kernel(self, op, kernel_name=None):
        """
        Sinh mã cho 1 Op đơn lẻ (phục vụ baseline Non-fused để benchmark so sánh).
        """
        kernel_name = kernel_name or f"kernel_{op.op_type}_{id(op) % 1000}"
        kernel_name = re.sub(r'[^a-zA-Z0-9_]', '_', kernel_name)

        in_vars = [f"in_{i}[idx]" for i in range(len(op.inputs))]
        out_var = "res"
        expr = op.get_cuda_expr(in_vars, out_var)

        params = [f"const float* __restrict__ in_{i}" for i in range(len(op.inputs))]
        params.append("float* __restrict__ out")
        params.append("int N")

        cuda_source = f"""extern "C" __global__
void {kernel_name}({', '.join(params)}) {{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    {expr}
    out[idx] = res;
}}
"""
        return kernel_name, cuda_source

