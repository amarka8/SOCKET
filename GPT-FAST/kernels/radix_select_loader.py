"""load_inline wrapper for radix_select.cu (DESIGN B: fused scorer + radix threshold select).

No --use_fast_math: the fused scorer must stay bit-identical to soft_hash_collision_kernel_3
(same fp32 adds in the same l order, then one multiply), and load_inline compiles that kernel
with default flags.
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
