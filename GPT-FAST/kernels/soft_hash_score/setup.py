from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="soft_hash_score_ext",
    ext_modules=[
        CUDAExtension(
            name="soft_hash_score_ext",
            sources=[
                "soft_hash_score_bindings.cpp",  # renamed from soft_hash_score.cpp
                "soft_hash_score.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
