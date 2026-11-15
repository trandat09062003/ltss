import numpy as np

class Op:
    """Lớp cơ sở đại diện cho một phép toán trong Đồ thị tính toán (DAG)."""
    op_type = "op"
    is_fusible = False  # Có thể gom cụm (kernel fusion) hay không

    def __init__(self, inputs, name=None):
        self.inputs = list(inputs)
        self.outputs = []
        self.name = name or f"{self.op_type}_{id(self) % 10000}"

    @classmethod
    def apply(cls, *args, **kwargs):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self, *out_grads):
        raise NotImplementedError

    def get_cuda_expr(self, in_vars, out_var, **kwargs):
        """Sinh chuỗi biểu thức CUDA C++ cho kernel fusion."""
        raise NotImplementedError


class AddOp(Op):
    op_type = "add"
    is_fusible = True  # Pointwise op -> Fusible

    @classmethod
    def apply(cls, a, b):
        from mini_compiler.core.tensor import Tensor
        if not isinstance(a, Tensor):
            a = Tensor(a)
        if not isinstance(b, Tensor):
            b = Tensor(b)

        op = cls([a, b])
        out_data = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad, creator=op, name=f"{op.name}_out")
        op.outputs = [out]
        return out

    def backward(self, out_grad):
        ga = out_grad if self.inputs[0].requires_grad else None
        gb = out_grad if self.inputs[1].requires_grad else None
        
        # Xử lý broadcasting nếu shape lệch nhau (e.g. vector bias + matrix)
        if ga is not None and ga.shape != self.inputs[0].shape:
            axes = tuple(range(ga.ndim - self.inputs[0].data.ndim))
            ga = ga.sum(axis=axes)
        if gb is not None and gb.shape != self.inputs[1].shape:
            axes = tuple(range(gb.ndim - self.inputs[1].data.ndim))
            gb = gb.sum(axis=axes)
            
        return [ga, gb]

    def get_cuda_expr(self, in_vars, out_var, **kwargs):
        return f"float {out_var} = {in_vars[0]} + {in_vars[1]};"


class MulOp(Op):
    op_type = "mul"
    is_fusible = True  # Pointwise op -> Fusible

    @classmethod
    def apply(cls, a, b):
        from mini_compiler.core.tensor import Tensor
        if not isinstance(a, Tensor):
            a = Tensor(a)
        if not isinstance(b, Tensor):
            b = Tensor(b)

        op = cls([a, b])
        out_data = a.data * b.data
        requires_grad = a.requires_grad or b.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad, creator=op, name=f"{op.name}_out")
        op.outputs = [out]
        return out

    def backward(self, out_grad):
        a, b = self.inputs
        ga = (out_grad * b.data) if a.requires_grad else None
        gb = (out_grad * a.data) if b.requires_grad else None
        return [ga, gb]

    def get_cuda_expr(self, in_vars, out_var, **kwargs):
        return f"float {out_var} = {in_vars[0]} * {in_vars[1]};"


class ReLUOp(Op):
    op_type = "relu"
    is_fusible = True  # Pointwise op -> Fusible

    @classmethod
    def apply(cls, a):
        from mini_compiler.core.tensor import Tensor
        if not isinstance(a, Tensor):
            a = Tensor(a)

        op = cls([a])
        if hasattr(a.data, '__class__') and 'cupy' in str(a.data.__class__):
            import cupy as cp
            out_data = cp.maximum(a.data, 0.0)
        else:
            out_data = np.maximum(a.data, 0.0)

        out = Tensor(out_data, requires_grad=a.requires_grad, creator=op, name=f"{op.name}_out")
        op.outputs = [out]
        return out

    def backward(self, out_grad):
        a = self.inputs[0]
        if not a.requires_grad:
            return [None]
        mask = (a.data > 0).astype(a.data.dtype)
        return [out_grad * mask]

    def get_cuda_expr(self, in_vars, out_var, **kwargs):
        return f"float {out_var} = ({in_vars[0]} > 0.0f) ? {in_vars[0]} : 0.0f;"


class DropoutOp(Op):
    op_type = "dropout"
    is_fusible = True  # Pointwise op -> Fusible

    def __init__(self, inputs, p=0.5, training=True, seed=42, name=None):
        super().__init__(inputs, name=name)
        self.p = p
        self.training = training
        self.seed = seed
        self.mask = None

    @classmethod
    def apply(cls, a, p=0.5, training=True, seed=42):
        from mini_compiler.core.tensor import Tensor
        if not isinstance(a, Tensor):
            a = Tensor(a)

        op = cls([a], p=p, training=training, seed=seed)
        if not training or p == 0.0:
            out_data = a.data
            op.mask = None
        else:
            scale = 1.0 / (1.0 - p)
            # Dùng np.random rồi copy lên GPU để không phụ thuộc vào thư viện libcurand.so
            rng = np.random.RandomState(seed)
            mask_np = (rng.rand(*a.shape) >= p).astype(np.float32) * scale
            if hasattr(a.data, '__class__') and 'cupy' in str(a.data.__class__):
                import cupy as cp
                op.mask = cp.asarray(mask_np)
            else:
                op.mask = mask_np
            out_data = a.data * op.mask

        out = Tensor(out_data, requires_grad=a.requires_grad, creator=op, name=f"{op.name}_out")
        op.outputs = [out]
        return out

    def backward(self, out_grad):
        a = self.inputs[0]
        if not a.requires_grad:
            return [None]
        if not self.training or self.mask is None:
            return [out_grad]
        return [out_grad * self.mask]

    def get_cuda_expr(self, in_vars, out_var, **kwargs):
        # Fast PRNG LCG sinh số ngẫu nhiên trực tiếp trong kernel thread
        p = self.p
        scale = 1.0 / (1.0 - p)
        return (
            f"// In-thread PRNG for Dropout\n"
            f"    unsigned int _seed = {self.seed}u + (unsigned int)idx;\n"
            f"    _seed = _seed * 1664525u + 1013904223u;\n"
            f"    float _rand_val = (float)(_seed & 0x00FFFFFF) / 16777216.0f;\n"
            f"    float {out_var} = (_rand_val >= {p}f) ? ({in_vars[0]} * {scale}f) : 0.0f;"
        )


class MatMulOp(Op):
    op_type = "matmul"
    is_fusible = False  # Compute-intensive GEMM -> Thường dùng cuBLAS hoặc Tiled Kernel

    @classmethod
    def apply(cls, a, b):
        from mini_compiler.core.tensor import Tensor
        if not isinstance(a, Tensor):
            a = Tensor(a)
        if not isinstance(b, Tensor):
            b = Tensor(b)

        op = cls([a, b])
        out_data = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad, creator=op, name=f"{op.name}_out")
        op.outputs = [out]
        return out

    def backward(self, out_grad):
        a, b = self.inputs
        ga = (out_grad @ b.data.T) if a.requires_grad else None
        gb = (a.data.T @ out_grad) if b.requires_grad else None
        return [ga, gb]
