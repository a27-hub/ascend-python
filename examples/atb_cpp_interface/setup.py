import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

import torch
import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension


BASE_DIR = Path(__file__).resolve().parent
PYTORCH_NPU_INSTALL_PATH = Path(torch_npu.__file__).resolve().parent
USE_NINJA = os.getenv("USE_NINJA") == "1"
EXTENSION_NAME = "atb_cpp_interface_extension"
CXX11_ABI = "1" if torch.compiled_with_cxx11_abi() else "0"


def find_first_file(search_root: Path, file_name: str):
    for path in search_root.rglob(file_name):
        if path.is_file():
            return path
    return None


def resolve_libop_plugin_atb():
    repo_root = BASE_DIR.parents[1]
    built_lib = find_first_file(repo_root, "libop_plugin_atb.so")
    if built_lib is not None:
        return built_lib

    installed_lib = PYTORCH_NPU_INSTALL_PATH / "lib" / "libop_plugin_atb.so"
    if installed_lib.is_file():
        return installed_lib

    raise RuntimeError(
        "libop_plugin_atb.so not found. "
        "Expected it under the torch_npu install root or the local repository build tree."
    )


def resolve_real_libatb():
    abi_dir = f"cxx_abi_{CXX11_ABI}"
    candidate_roots = [
        os.getenv("ATB_HOME_PATH"),
        os.getenv("NNAL_HOME_PATH"),
        os.getenv("ASCEND_NNAL_HOME"),
        "/home/cann_custom_path/20260415/nnal/atb/9.0.0/atb",
    ]
    for root in candidate_roots:
        if not root:
            continue
        root_path = Path(root)
        if not root_path.exists():
            continue
        abi_root = root_path / abi_dir
        if abi_root.exists():
            found = find_first_file(abi_root, "libatb.so")
            if found is not None:
                return found
        found = find_first_file(root_path, "libatb.so")
        if found is not None and abi_dir in str(found):
            return found
    return None


libop_plugin_atb = resolve_libop_plugin_atb()
libop_plugin_atb_dir = libop_plugin_atb.parent
libatb = resolve_real_libatb()
if libatb is None:
    raise RuntimeError(
        "real libatb.so not found. "
        "Please install NNAL/ATB and set ATB_HOME_PATH, NNAL_HOME_PATH or ASCEND_NNAL_HOME."
    )

runtime_rpaths = [str(libop_plugin_atb_dir), str(libatb.parent)]
extra_link_args = [f"-Wl,-rpath,{runtime_path}" for runtime_path in runtime_rpaths]
library_dirs = [str(libop_plugin_atb_dir), str(libatb.parent)]
libraries = ["op_plugin_atb", "atb"]

ext_modules = [
    NpuExtension(
        name=EXTENSION_NAME,
        sources=[str(BASE_DIR / "csrc" / "atb_quantize_per_tensor_extension.cpp")],
        extra_compile_args=[
            "-I" + str(PYTORCH_NPU_INSTALL_PATH / "include" / "third_party" / "acl" / "inc"),
            "-I" + str(PYTORCH_NPU_INSTALL_PATH / "include" / "third_party" / "op-plugin"),
            "-I" + str(PYTORCH_NPU_INSTALL_PATH / "include" / "third_party" / "op-plugin" / "op_plugin" / "include"),
        ],
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name=EXTENSION_NAME,
    version="1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
