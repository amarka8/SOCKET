"""load_inline wrapper for rowpack_scorer.cu (head-packed GQA4 soft-hash scorer).

Compiled with default arithmetic flags -- no --use_fast_math: the kernel's fp32 adds must
stay bit-identical to the Triton scorer's.
"""
import os

from torch.utils.cpp_extension import load_inline

_EXT = None


def load_rowpack_scorer(*, verbose: bool = False):
    """Compile / load the packed-scorer extension. Cached per process."""
    global _EXT
    if _EXT is not None:
        return _EXT
    src_path = os.path.join(os.path.dirname(__file__), "rowpack_scorer.cu")
    with open(src_path, "r", encoding="utf-8") as f:
        cuda_src = f.read()
    _EXT = load_inline(
        name="socket_rowpack_scorer_ext",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=None,
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=verbose,
    )
    return _EXT
