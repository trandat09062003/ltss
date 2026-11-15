import os
import sys

HAS_CUDA = False
try:
    import cupy as cp
    from cupy.cuda import nvrtc
    from cupy.cuda import runtime as cuda_runtime
    from cupy.cuda import function as cuda_function
    if cp.cuda.is_available():
        HAS_CUDA = True
except Exception:
    HAS_CUDA = False


class NVRTCCompiler:
    """
    Quản lý việc biên dịch Just-In-Time (JIT) mã nguồn CUDA C++ bằng NVRTC
    và nạp kernel vào GPU.
    """
    def __init__(self):
        self.has_cuda = HAS_CUDA
        self._compiled_cache = {}

    def compile(self, cuda_source, kernel_name, arch=None):
        """
        Biên dịch chuỗi mã CUDA C++ thành kernel thực thi trên GPU.
        Sử dụng backend NVRTC của CuPy để tự động tương thích với phiên bản Driver/PTX.
        """
        cache_key = (cuda_source, kernel_name)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        if not self.has_cuda:
            print("[NVRTCCompiler] Không phát hiện GPU NVIDIA/CUDA. Chuyển sang chế độ mô phỏng (CPU Fallback).")
            return self._create_cpu_fallback(cuda_source, kernel_name)

        # 1. Dùng cp.RawKernel (NVRTC backend tối ưu, tự động giải quyết tương thích PTX/Driver)
        try:
            kernel_func = cp.RawKernel(cuda_source, kernel_name, options=("--use_fast_math", "-std=c++14"))
            self._compiled_cache[cache_key] = kernel_func
            return kernel_func
        except Exception:
            pass

        # 2. Fallback sang NVRTC CUBIN trực tiếp (dùng sm_86 cho RTX 3060 để tránh lỗi PTX ISA)
        try:
            prog = nvrtc.createProgram(cuda_source, f"{kernel_name}.cu", [], [])
            device_id = cuda_runtime.getDevice()
            props = cuda_runtime.getDeviceProperties(device_id)
            major = props['major']
            minor = props['minor']
            options = ["--use_fast_math", "-std=c++14", f"--gpu-architecture=sm_{major}{minor}"]
            
            nvrtc.compileProgram(prog, options)
            cubin = nvrtc.getCUBIN(prog)
            nvrtc.destroyProgram(prog)

            module = cuda_function.Module()
            module.load(cubin)
            kernel_func = module.get_function(kernel_name)
            self._compiled_cache[cache_key] = kernel_func
            return kernel_func
        except Exception as e:
            # Cuối cùng thử PTX
            raise RuntimeError(f"NVRTC JIT compilation failed for {kernel_name}: {e}")

    def _create_cpu_fallback(self, cuda_source, kernel_name):
        """Chế độ giả lập khi chạy trên máy không có card NVIDIA."""
        def simulated_kernel(grid, block, args):
            pass
        return simulated_kernel


class CUDAExecutor:
    """
    Thực thi Kernel CUDA và hỗ trợ đo đạc thời gian chính xác bằng CUDA Events.
    """
    def __init__(self, compiler=None):
        self.compiler = compiler or NVRTCCompiler()

    def launch(self, kernel_func, grid_dim, block_dim, args, stream=None):
        """
        Gọi thực thi kernel trên GPU: kernel<<<grid_dim, block_dim, 0, stream>>>(*args).
        """
        if not HAS_CUDA:
            return None
        return kernel_func(grid_dim, block_dim, args, stream=stream)

    def benchmark_kernel(self, kernel_func, grid_dim, block_dim, args, warmup=10, repeat=100):
        """
        Đo thời gian thực thi của kernel bằng CUDA Events (độ chính xác microsecond).
        """
        if not HAS_CUDA:
            return 0.0

        for _ in range(warmup):
            kernel_func(grid_dim, block_dim, args)
        cp.cuda.Stream.null.synchronize()

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        for _ in range(repeat):
            kernel_func(grid_dim, block_dim, args)
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event) / repeat
        return elapsed_ms
