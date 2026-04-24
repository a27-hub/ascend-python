import glob
import os
import torch_npu
from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


BASE_DIR = Path(__file__).resolve().parent
PYTORCH_NPU_INSTALL_PATH = Path(torch_npu.__file__).resolve().parent
USE_NINJA = os.getenv("USE_NINJA") == "1"


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
    atb_home = os.getenv("ATB_HOME_PATH")
    if atb_home:
        root_path = Path(atb_home)
        if root_path.exists():
            found = find_first_file(root_path, "libatb.so")
            if found is not None:
                return found
    raise RuntimeError(
        "real libatb.so not found. Please set ATB_HOME_PATH to a directory that contains libatb.so."
    )


libop_plugin_atb = resolve_libop_plugin_atb()
libatb = resolve_real_libatb()
source_files = glob.glob(str(BASE_DIR / "csrc" / "*.cpp"))

ext_modules = [
    NpuExtension(
        name="atb_cpp_interface._C",
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
    name="atb_cpp_interface",
    version="1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
