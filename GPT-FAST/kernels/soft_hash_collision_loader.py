import os
from string import Template

import torch
from torch.utils.cpp_extension import load_inline

CUDA_HEADERS = """
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAStream.h>
    #include <c10/cuda/CUDAException.h>
    #include <stdint.h>
"""

CUDA_WRAPPER_TEMPLATE = Template(
    """
    torch::Tensor soft_hash_collision(
        torch::Tensor q_probs,
        torch::Tensor key_buckets,
        torch::Tensor allowed_ext,
        torch::Tensor v_hist_in
    ) {
        TORCH_CHECK(q_probs.is_cuda(), "q_probs must be CUDA");
        TORCH_CHECK(key_buckets.is_cuda(), "key_buckets must be CUDA");
        TORCH_CHECK(allowed_ext.is_cuda(), "allowed_ext must be CUDA");
        TORCH_CHECK(v_hist_in.is_cuda(), "v_hist must be CUDA");

        TORCH_CHECK(q_probs.is_contiguous(), "q_probs must be contiguous");
        TORCH_CHECK(key_buckets.is_contiguous(), "key_buckets must be contiguous");
        TORCH_CHECK(allowed_ext.is_contiguous(), "allowed_ext must be contiguous");
        TORCH_CHECK(v_hist_in.is_contiguous(), "v_hist must be contiguous");

        TORCH_CHECK(q_probs.scalar_type() == torch::kFloat, "q_probs must be float32");
        // int16 is REQUIRED: pack_bits over K=8 bits yields bucket values 0..255 (R=256),
        // which DO NOT fit in signed int8 (max 127) -> values 128..255 would wrap negative
        // and mis-index q_probs[...,r]. int8 buckets are a correctness bug; reject loudly.
        TORCH_CHECK(key_buckets.scalar_type() == torch::kInt16, "key_buckets must be int16 (int8 loses buckets 128..255 for R=256)");
        TORCH_CHECK(allowed_ext.scalar_type() == torch::kBool, "allowed_ext must be bool");
        TORCH_CHECK(v_hist_in.scalar_type() == torch::kFloat, "v_hist must be float32");

        TORCH_CHECK(q_probs.dim() == 5, "q_probs must be [B,H,1,L,R]");
        TORCH_CHECK(key_buckets.dim() == 4, "key_buckets must be [B,H,L,T_k]");
        TORCH_CHECK(allowed_ext.dim() == 4, "allowed_ext must be [B,H,1,T_k]");
        TORCH_CHECK(v_hist_in.dim() == 4, "v_hist must be [B,H,1,T_k]");

        auto q = q_probs.contiguous();
        auto kb = key_buckets.contiguous();
        auto al = allowed_ext.contiguous();
        auto v_hist = v_hist_in.contiguous();

        int64_t B = q.size(0);
        int64_t H = q.size(1);
        int64_t L = q.size(3);
        int64_t R = q.size(4);
        int64_t T_k = kb.size(3);
        int64_t Hkv = kb.size(1);   // PER-KV-HEAD: key_buckets is [B,Hkv,L,T_k]

        TORCH_CHECK(H % Hkv == 0, "n_head must be divisible by n_kv_heads");
        TORCH_CHECK(kb.size(0) == B && kb.size(2) == L,
                    "key_buckets shape mismatch");
        TORCH_CHECK(al.size(0) == B && al.size(1) == H && al.size(2) == 1 && al.size(3) == T_k,
                    "allowed_ext shape mismatch");
        TORCH_CHECK(v_hist.size(0) == B && v_hist.size(1) == Hkv && v_hist.size(2) == 1 && v_hist.size(3) == T_k,
                    "v_hist shape mismatch");

        auto out = torch::zeros({B, H, 1, T_k}, q.options());

        $torch_checks
        $configuration
        $launch

        return out;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("soft_hash_collision", &soft_hash_collision,
              "Soft hash collision (q_probs, key_buckets, allowed_ext, v_hist -> out)");
    }
    """
)


def get_kernel_code(version: int, filename: str) -> tuple[str, str, str, str]:
    kernel_name = f"soft_hash_collision_kernel_{version}"
    start_marker = f"start: {kernel_name}"
    end_marker = f"end: {kernel_name}"

    def _clean_section(section: str) -> str:
        lines = []
        for line in section.splitlines():
            cleaned = line.lstrip(" \t*").rstrip()
            if cleaned:
                lines.append(cleaned)
        return "\n".join(lines).strip()

    def _extract_section(docstring: str, header: str, *, required: bool, next_headers: tuple[str, ...]) -> str:
        header_idx = docstring.find(header)
        if header_idx == -1:
            if required:
                raise ValueError(f"{header} block for {kernel_name} not found")
            return ""

        start_idx = header_idx + len(header)
        end_idx = len(docstring)

        for next_header in next_headers:
            next_idx = docstring.find(next_header, start_idx)
            if next_idx != -1 and next_idx < end_idx:
                end_idx = next_idx

        section = _clean_section(docstring[start_idx:end_idx])
        if required and not section:
            raise ValueError(f"{header} block for {kernel_name} is empty")
        return section

    with open(filename, "r", encoding="utf-8") as f:
        source = f.read()

    start = source.find(start_marker)
    if start == -1:
        raise ValueError(f"Start marker for {kernel_name} not found in {filename}")

    end_marker_idx = source.find(end_marker, start)
    if end_marker_idx == -1:
        raise ValueError(f"End marker for {kernel_name} not found in {filename}")

    doc_start = source.find("/*", start)
    if doc_start == -1 or doc_start > end_marker_idx:
        raise ValueError(f"Docstring for {kernel_name} not found between markers")

    doc_end = source.find("*/", doc_start)
    if doc_end == -1 or doc_end > end_marker_idx:
        raise ValueError(f"Docstring for {kernel_name} not properly closed")

    docstring = source[doc_start + 2:doc_end]
    configuration = _extract_section(docstring, "Configuration:", required=True,
                                     next_headers=("Launch:", "Torch Checks:"))
    launch = _extract_section(docstring, "Launch:", required=True,
                              next_headers=("Torch Checks:",))
    torch_checks = _extract_section(docstring, "Torch Checks:", required=False,
                                    next_headers=())

    kernel_src = source[doc_end + 2:end_marker_idx]

    return kernel_src, configuration, launch, torch_checks


# ---------------------------------------------------------------------------
# SOCKET_SCORER_STREAMFIX  (default "1" == FIXED;  "0" reproduces the bug)
#
# THE BUG. The .cu docstring's `Launch:` block is
#     soft_hash_collision_kernel_3<<<grid, block>>>(...)
# i.e. no stream argument, so the kernel goes to the LEGACY DEFAULT STREAM and the launch is
# never error-checked. Under torch.compile(mode="reduce-overhead") the whole decode step is
# CAPTURED into a CUDA graph on a non-default (capture) stream. A legacy-default-stream
# launch is NOT captured as a graph node, so at every REPLAY this kernel DOES NOT RUN and
# `out` stays at the torch::zeros the wrapper allocated. Consequence: every compiled SOCKET
# number produced before this fix selected its heavy tokens from an ALL-ZERO score array --
# it got its single most expensive stage for free, and the selection was arbitrary.
#
# THE FIX. Launch on at::cuda::getCurrentCUDAStream() (the stream being captured) and check
# the launch. Measured cost at 140K: 93.4 -> 89.3 tok/s at L=10, 94.1 -> 71.8 at L=50 with
# the Triton scorer; 93.4 -> 79.1 / 94.1 -> 59.0 with this CUDA scorer. Those post-fix
# numbers are the only honest ones.
#
# The two variants get DISTINCT load_inline extension names so the JIT BUILD cache (which
# keys on the name + sources) cannot hand a stale .so of one variant to the other.
# ---------------------------------------------------------------------------
_STREAM_FIX_NEEDLE = "<<<grid, block>>>"


def _streamfix_enabled() -> bool:
    return os.environ.get("SOCKET_SCORER_STREAMFIX", "1") != "0"


def load_soft_hash_collision(version: int = 3, *, verbose: bool = False):
    kernels_path = os.path.join(os.path.dirname(__file__), "soft_hash_collision.cu")
    code, configuration, launch, torch_checks = get_kernel_code(version, kernels_path)

    suffix = ""
    if _streamfix_enabled():
        if _STREAM_FIX_NEEDLE not in launch:
            raise ValueError(
                f"cannot apply the scorer stream fix: {_STREAM_FIX_NEEDLE!r} not found in the "
                f"Launch: block of soft_hash_collision.cu (did the launch syntax change?)")
        launch = launch.replace(
            _STREAM_FIX_NEEDLE,
            "<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>", 1)
        launch = launch + "\n        C10_CUDA_KERNEL_LAUNCH_CHECK();"
        suffix = "_cs"

    wrapper = CUDA_WRAPPER_TEMPLATE.substitute(
        configuration=configuration,
        launch=launch,
        torch_checks=torch_checks,
    )

    cuda_sources = CUDA_HEADERS + "\n" + code + "\n" + wrapper

    ext = load_inline(
        name=f"soft_hash_collision_ext_v{version}{suffix}",
        cpp_sources="",
        cuda_sources=cuda_sources,
        functions=None,
        verbose=verbose,
    )
    return ext
