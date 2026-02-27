import os
import sys
import sysconfig
import subprocess
from setuptools import setup, Extension, find_packages
import torch
import torch_npu
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 获取PyTorch版本
PYTORCH_VERSION = subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.__version__.split("+")[0])']).decode('utf-8').strip()
version_parts = PYTORCH_VERSION.split('.')
PYTORCH_VERSION_DIR = f"v{version_parts[0]}r{version_parts[1]}"

#设置环境变量
os.environ["PYTORCH_VERSION"] = PYTORCH_VERSION
os.environ["PYTORCH_CUSTOM_DERIVATIVES_PATH"] = os.path.join(os.path.dirname(__file__), f"op-plugin/config/{PYTORCH_VERSION_DIR}/derivatives.yaml")
os.environ["ACNN_EXTENSION_PATH"] = os.path.dirname(__file__)
os.environ["ACNN_EXTENSION_SWITCH"] = "TRUE"

#获取所有需要编译的源文件
def get_sources():
    sources = []
    # 添加csrc/aten目录下的源文件
    aten_dir = os.path.join(os.path.dirname(__file__), "torch_npu/csrc/aten")
    if os.path.exists(aten_dir):
        for root, _, files in os.walk(aten_dir):
            for file in files:
                if file.endswith(".cpp") or file.endswith(".cc"):
                    sources.append(os.path.join(root, file))
    # 添加op-plugin/ops目录下的源文件
    ops_dir = os.path.join(os.path.dirname(__file__), "op_plugin")
    if os.path.exists(ops_dir):
        for root, _, files in os.walk(ops_dir):
            for file in files:
                if file.endswith(".cpp") or file.endswith(".cc"):
                    sources.append(os.path.join(root, file))

    BUILD_EXCLUDE_LIST = [f"{aten_dir}/VariableTypeEverything.cpp",
        f"{aten_dir}/ADInplaceOrViewTypeEverything.cpp",
        f"{aten_dir}/python_functionsEverything.cpp",
        f"{aten_dir}/RegisterFunctionalizationEverything.cpp"]

    sources_new = [cur_file for cur_file in sources if cur_file not in BUILD_EXCLUDE_LIST]
    print("====sources_new:", sources_new)

    return sources_new

#获取所有需要的头文件目录
def get_include_dirs():
    PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))

    include_dirs = []
    # 添加csrc/aten目录
    aten_dir = os.path.join(os.path.dirname(__file__), "torch_npu/csrc/aten")
    if os.path.exists(aten_dir):
        include_dirs.append(aten_dir)
    # 添加op-plugin/ops目录
    ops_dir = os.path.join(os.path.dirname(__file__), "op_plugin")
    if os.path.exists(ops_dir):
        include_dirs.append(ops_dir)
    # # 添加utils目录
    # utils_dir = os.path.join(os.path.dirname(__file__), "utils")
    # if os.path.exists(utils_dir):
    #     include_dirs.append(utils_dir)

    base_dir = os.path.dirname(__file__)
    if os.path.exists(base_dir):
        include_dirs.append(base_dir)

    torch_npu_dir = PYTORCH_NPU_INSTALL_PATH
    include_dirs.append(os.path.join(torch_npu_dir, 'include'))
    include_dirs.append(os.path.join(torch_npu_dir, 'include', 'third_party', 'acl', 'inc'))
    include_dirs.append(os.path.join(torch_npu_dir, 'include', 'third_party', 'hccl', 'inc'))
    include_dirs.append(os.path.join(torch_npu_dir, 'include', 'third_party', 'op-plugin'))
    return include_dirs

#获取所有编译参数
def get_compile_args():
    compile_args = ["-std=c++17"]
    # 如果是Windows平台
    if sys.platform == "win32":
        compile_args.append("/MD")
    # 如果是Linux平台
    elif sys.platform == "linux":
        compile_args.append("-fPIC")
    return compile_args

def get_dependency_paths():
    # python_include = sysconfig.get_config_var("INCLUDEPY")
    python_lib = sysconfig.get_config_var("LIBDIR")

    # torch_include_paths = cpp_extension.include_paths()
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

    torch_npu_path = os.path.dirname(torch_npu.__file__)
    # torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib = os.path.join(torch_npu_path, "lib")

    all_libs = list([
        python_lib,
        torch_lib,
        torch_npu_lib,
    ])

    return {
        # "all_includes": all_include_paths,
        "all_libs": all_libs
    }

#获取所有链接参数
def get_link_args():
    link_args = []

    link_args.append("-ltorch_npu")
    link_args.append("-ltorch")
    link_args.append("-lc10")

    dep_paths = get_dependency_paths()
    for lib_dir in dep_paths["all_libs"]:
        link_args.append(f"-L{lib_dir}")
    return link_args

#配置扩展
#使用PyTorch的CppExtension替代标准Extension，提供更好的PyTorch集成
extensions = [
    CppExtension(
        "aclnn_extension.custom_aclnn_extension_lib",
        sources=get_sources(),
        include_dirs=get_include_dirs(),
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args(),
    )
]

#安装配置
setup(
    name="aclnn_extension",
    version="1.0.0",
    author="",
    author_email="",
    description="ACLNN extension for PyTorch",
    long_description="",
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension,
    },
    zip_safe=False,
    install_requires=[
        f"torch=={PYTORCH_VERSION}"
    ],
    packages=find_packages()
)
