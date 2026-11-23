# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""
Pluggable Kernel Backend Registry untuk `mshqc`:
  - `"cpp"` (default): Kernel C++20 OpenMP/BLAS/TBLIS terkompilasi (`mshqc._mshqc.kernels`, 0 overhead)
  - `"numpy"`: Implementasi referensi murni NumPy/SciPy (untuk eksperimen, AD, atau basis GPU backend)
  - Kustom: Daftarkan backend baru (misal PyTorch/JAX/ROCm) via `register_backend(name, obj)`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List
from .._mshqc import kernels as _cpp_kernels
from .numpy_backend import NumpyKernelBackend

_REGISTRY: Dict[str, Any] = {
    "cpp": _cpp_kernels,
    "numpy": NumpyKernelBackend(),
}
_ACTIVE_BACKEND: str = "cpp"


def available_backends() -> List[str]:
    return sorted(_REGISTRY.keys())


def register_backend(name: str, backend: Any) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Nama backend tidak boleh kosong.")
    _REGISTRY[key] = backend


def set_backend(name: str) -> None:
    global _ACTIVE_BACKEND
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Backend {name!r} tidak terdaftar. Pilihan: {available_backends()}")
    _ACTIVE_BACKEND = key


def get_backend(name: str = "") -> Any:
    key = (name or _ACTIVE_BACKEND).strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Backend {name!r} tidak terdaftar. Pilihan: {available_backends()}")
    return _REGISTRY[key]


@contextmanager
def use_backend(name: str) -> Iterator[Any]:
    prev = _ACTIVE_BACKEND
    set_backend(name)
    try:
        yield get_backend(name)
    finally:
        set_backend(prev)




class _ActiveKernelProxy:
    """Proxy zero-boilerplate yang meneruskan pemanggilan kernel ke backend aktif."""
    # [INJECTED_V2_ENGINE_FALLBACK_PROXY]

    def __getattr__(self, name: str) -> Any:
        backend = _REGISTRY[_ACTIVE_BACKEND]
        if hasattr(backend, name):
            return getattr(backend, name)
        if getattr(backend, "_fallback_to_cpp", False) and hasattr(_cpp_kernels, name):
            return getattr(_cpp_kernels, name)
        raise AttributeError(
            f"Kernel {name!r} tidak tersedia di backend {_ACTIVE_BACKEND!r}."
        )

    def __repr__(self) -> str:
        return f"<mshqc.ActiveKernelProxy active={_ACTIVE_BACKEND!r}>"


active_kernels = _ActiveKernelProxy()


def enable_torch_backend(device: str = "auto", name: str = "torch", fallback_to_cpp: bool = True):
    """Daftarkan TorchKernelBackend (CUDA FP64 / CPU) ke registry tanpa mengganggu default."""
    from .torch_backend import TorchKernelBackend
    backend = TorchKernelBackend(device=device, fallback_to_cpp=fallback_to_cpp)
    register_backend(name, backend)
    return backend


__all__ = [
    "NumpyKernelBackend",
    "available_backends",
    "register_backend",
    "set_backend",
    "get_backend",
    "use_backend",
    "active_kernels",
    "enable_torch_backend",
]
