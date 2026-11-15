import numpy as np

class Tensor:
    """
    Tensor cơ bản hỗ trợ lưu trữ dữ liệu, đồ thị tính toán (DAG) và Autograd.
    Tương thích cả Numpy (chạy thử CPU) và CuPy (chạy GPU/CUDA).
    """
    def __init__(self, data, requires_grad=False, creator=None, name=None):
        if not isinstance(data, (np.ndarray, list, float, int)):
            # CuPy array check duck-typing (nếu có cupy)
            if not hasattr(data, 'shape'):
                raise TypeError(f"Unsupported data type: {type(data)}")
        
        if isinstance(data, (list, float, int)):
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = data
            
        self.requires_grad = requires_grad
        self.creator = creator  # Op/Node đã tạo ra tensor này
        self.grad = None
        self.name = name or f"t_{id(self) % 10000}"

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def backward(self, grad=None):
        """
        Kích hoạt Reverse-mode Autodiff từ tensor này ngược về các leaf tensors.
        """
        if not self.requires_grad:
            return

        if grad is None:
            # Mặc định dLoss / dLoss = 1.0
            if hasattr(self.data, '__class__') and 'cupy' in str(self.data.__class__):
                import cupy as cp
                grad = cp.ones_like(self.data)
            else:
                grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

        # Sắp xếp tô-pô (Topological Sort) từ đồ thị
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                if tensor.creator is not None:
                    for inp in tensor.creator.inputs:
                        build_topo(inp)
                    topo.append(tensor.creator)

        build_topo(self)

        # Duyệt ngược theo thứ tự tô-pô để lan truyền gradient
        for op in reversed(topo):
            # Lấy grad của các output
            out_grads = [out.grad for out in op.outputs]
            # Gọi backward của op
            in_grads = op.backward(*out_grads)
            if not isinstance(in_grads, (list, tuple)):
                in_grads = [in_grads]

            for inp, g in zip(op.inputs, in_grads):
                if inp.requires_grad and g is not None:
                    if inp.grad is None:
                        inp.grad = g
                    else:
                        inp.grad = inp.grad + g

    # --- Operator Overloads ---
    def __add__(self, other):
        from mini_compiler.core.ops import AddOp
        return AddOp.apply(self, other)

    def __radd__(self, other):
        from mini_compiler.core.ops import AddOp
        return AddOp.apply(other, self)

    def __mul__(self, other):
        from mini_compiler.core.ops import MulOp
        return MulOp.apply(self, other)

    def __rmul__(self, other):
        from mini_compiler.core.ops import MulOp
        return MulOp.apply(other, self)

    def __matmul__(self, other):
        from mini_compiler.core.ops import MatMulOp
        return MatMulOp.apply(self, other)

    def relu(self):
        from mini_compiler.core.ops import ReLUOp
        return ReLUOp.apply(self)

    def dropout(self, p=0.5, training=True, seed=42):
        from mini_compiler.core.ops import DropoutOp
        return DropoutOp.apply(self, p=p, training=training, seed=seed)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, name='{self.name}')"

