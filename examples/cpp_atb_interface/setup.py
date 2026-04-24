import glob
import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension

import torch
import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension


BASE_DIR = Path(__file__).resolve().parent
PYTORCH_NPU_INSTALL_PATH = Path(torch_npu.__file__).resolve().parent
USE_NINJA = os.getenv("USE_NINJA") == "1"
CXX11_ABI = "1" if torch.compiled_with_cxx11_abi() else "0"


def find_first_file(search_root: Path, file_name: str):
    for path in search_root.rglob(file_name):
        if path.is_file():
            return path
    return None


def resolve_libop_plugin_atb():
    installed_lib = PYTORCH_NPU_INSTALL_PATH / "lib" / "libop_plugin_atb.so"
    if installed_lib.is_file():
        return installed_lib

    raise RuntimeError("libop_plugin_atb.so not found under the torch_npu install root.")


def resolve_real_libatb():
    candidate_roots = [
        os.getenv("ATB_HOME_PATH"),
        os.getenv("NNAL_HOME_PATH"),
        os.getenv("ASCEND_NNAL_HOME"),
    ]
    for root in candidate_roots:
        if not root:
            continue
        root_path = Path(root)
        if not root_path.exists():
            continue
        found = find_first_file(root_path, "libatb.so")
        if found is not None:
            return found
    raise RuntimeError(
        "real libatb.so not found. Please install NNAL/ATB and set ATB_HOME_PATH, NNAL_HOME_PATH or ASCEND_NNAL_HOME."
    )


libop_plugin_atb = resolve_libop_plugin_atb()
libatb = resolve_real_libatb()
source_files = glob.glob(str(BASE_DIR / "csrc" / "*.cpp"))

ext_modules = [
    NpuExtension(
        name="cpp_atb_interface._C",
        sources=source_files,
        extra_compile_args=[
            "-I" + str(PYTORCH_NPU_INSTALL_PATH / "include" / "third_party" / "acl" / "inc"),
            "-I" + str(PYTORCH_NPU_INSTALL_PATH / "include" / "third_party" / "op-plugin"),
            "-I" + str(PYTORCH_NPU_INSTALL_PATH / "include" / "third_party" / "op-plugin" / "op_plugin" / "include"),
        ],
        library_dirs=[str(libop_plugin_atb.parent), str(libatb.parent)],
        libraries=["op_plugin_atb", "atb"],
        extra_link_args=[
            f"-Wl,-rpath,{libop_plugin_atb.parent}",
            f"-Wl,-rpath,{libatb.parent}",
        ],
    )
]

setup(
    name="cpp_atb_interface",
    version="1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
