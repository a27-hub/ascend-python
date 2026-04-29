import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension


BASE_DIR = Path(__file__).resolve().parent
PYTORCH_NPU_INSTALL_PATH = Path(torch_npu.__file__).resolve().parent
USE_NINJA = os.getenv("USE_NINJA") == "1"
EXTENSION_NAME = "atb_quantize_per_tensor_extension"


def find_first_file(search_root: Path, file_name: str):
    for path in search_root.rglob(file_name):
        if path.is_file():
            return path
    return None


def resolve_libop_plugin_atb():
    installed_lib = PYTORCH_NPU_INSTALL_PATH / "lib" / "libop_plugin_atb.so"
    if installed_lib.is_file():
        return installed_lib

    repo_root = BASE_DIR.parents[1]
    built_lib = find_first_file(repo_root, "libop_plugin_atb.so")
    if built_lib is not None:
        return built_lib

    raise RuntimeError(
        "libop_plugin_atb.so not found. "
        "Expected it under the torch_npu install root or the local repository build tree."
    )


def resolve_optional_library(search_roots, file_name: str):
    for root in search_roots:
        if root is None:
            continue
        root = Path(root)
        if not root.exists():
            continue
        found = find_first_file(root, file_name)
        if found is not None:
            return found
    return None


libop_plugin_atb = resolve_libop_plugin_atb()
libop_plugin_atb_dir = libop_plugin_atb.parent
repo_root = BASE_DIR.parents[1]
libatb = resolve_optional_library(
    [
        PYTORCH_NPU_INSTALL_PATH,
        repo_root,
        os.getenv("ATB_HOME_PATH"),
        os.getenv("NNAL_HOME_PATH"),
    ],
    "libatb.so",
)

runtime_rpaths = [str(libop_plugin_atb_dir)]
if libatb is not None:
    runtime_rpaths.append(str(libatb.parent))

extra_link_args = [f"-Wl,-rpath,{runtime_path}" for runtime_path in runtime_rpaths]

ext_modules = [
    NpuExtension(
        name=EXTENSION_NAME,
        sources=[str(BASE_DIR / "csrc" / "atb_quantize_per_tensor_extension.cpp")],
        extra_compile_args=[
            "-I" + str(PYTORCH_NPU_INSTALL_PATH / "include" / "third_party" / "acl" / "inc"),
            "-I" + str(PYTORCH_NPU_INSTALL_PATH / "include" / "third_party" / "op-plugin"),
            "-I" + str(PYTORCH_NPU_INSTALL_PATH / "include" / "third_party" / "op-plugin" / "op_plugin" / "include"),
        ],
        library_dirs=[str(libop_plugin_atb_dir)],
        libraries=["op_plugin_atb"],
        extra_link_args=extra_link_args,
    ),
]

setup(
    name=EXTENSION_NAME,
    version="1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
