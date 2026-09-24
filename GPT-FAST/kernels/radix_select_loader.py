"""load_inline wrapper for radix_select.cu (exact top-M index select).

Compiled with default arithmetic flags -- no --use_fast_math. Nothing in this kernel does
floating-point arithmetic (it only reinterprets score bits through a monotone integer key),
but the flag would also apply to anything added later, and this file is on the selection path
where an inexact comparison would silently change which tokens are chosen.
"""
import os

from torch.utils.cpp_extension import load_inline

_EXT = None


def load_radix_select(*, verbose: bool = False):
    """Compile / load the radix-select extension. Cached per process."""
    global _EXT
    if _EXT is not None:
        return _EXT
    src_path = os.path.join(os.path.dirname(__file__), "radix_select.cu")
    with open(src_path, "r", encoding="utf-8") as f:
        cuda_src = f.read()
    _EXT = load_inline(
        name="socket_radix_select_ext",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=None,
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=verbose,
    )
    return _EXT
