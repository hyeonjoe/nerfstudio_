import pathlib, os, glob, os.path as osp, platform, sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ────────────────────────────── ① 소스 목록 ──────────────────────────────
extensions_dir1 = pathlib.Path(__file__).parent.resolve()
extensions_dir = osp.join(extensions_dir1, "cuda")
sources  = glob.glob(osp.join(extensions_dir, "csrc", "*.cu"))
sources += glob.glob(osp.join(extensions_dir, "csrc", "*.cpp"))
# 필요하다면 주석 해제 → ext.cpp 의 실제 위치가 맞는지 확인!
# sources += [osp.join(extensions_dir, "csrc", "ext.cpp")]
print("Sources files to compile:")
for src in sources:
    print(" -", src)
print(f"Total source files found: {len(sources)}")
# ────────────────────────────── ② 컴파일 옵션 ──────────────────────────────
extra_compile_args = {
    "cxx" : ["-O3", "-Wno-sign-compare"],
    "nvcc": ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr"]
}
if sys.platform == "darwin" and platform.machine() == "arm64":
    extra_compile_args["cxx"]  += ["-arch", "arm64"]
    extra_compile_args["nvcc"] += ["-arch", "arm64"]

current_dir = pathlib.Path(__file__).parent.resolve()
glm_path = osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")
include_dirs = [glm_path, osp.join(current_dir, "gsplat", "cuda", "include")]

# ────────────────────────────── ③ Extension 객체 ──────────────────────────
ext_modules = [
    CUDAExtension(
        name="csrc_custom",          # ★ 새 .so 이름
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=["-s"],             # strip 심벌(옵션)
    )
]

# ────────────────────────────── ④ setup : 빌드 전용 ────────────────────────
setup(
    name="gsplat_build_only",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True,
                                                       use_ninja=True)},
)

