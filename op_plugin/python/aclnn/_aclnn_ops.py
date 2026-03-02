import os
import sys
import types
import pathlib
from functools import wraps, lru_cache
import torch
import torch_npu

OVERADPT_API_LIST = [
    'matmul',
    'matmul.out',
    'matmul_backward',
    'matmul_double_backward',
]

OVERADPT_MODULE_NAME = 'overadpt'
OVERADPT_MODULE = types.ModuleType(OVERADPT_MODULE_NAME)

NNAL_EX = None
GLOBAL_E = None


def _add_overadpt_module():
    setattr(torch_npu, OVERADPT_MODULE_NAME, OVERADPT_MODULE)
    sys.modules[f'torch_npu.{OVERADPT_MODULE_NAME}'] = OVERADPT_MODULE


@lru_cache(None)
def _register_overadpt_extensions():
    global NNAL_EX, GLOBAL_E
    if NNAL_EX is not None:
        raise NNAL_EX from GLOBAL_E
    _patch_overadpt_ops()


def lazy_load_overadpt_so(api_func):
    @wraps(api_func)
    def wrapper(*args, **kwargs):
        _register_overadpt_extensions()
        return api_func(*args, **kwargs)

    return wrapper


def create_lazy_overadpt_function(api_name):
    @lazy_load_overadpt_so
    def generated_function(*args, **kwargs):
        return getattr(torch.ops.overadpt, api_name)(*args, **kwargs)

    generated_function.__name__ = api_name
    return generated_function


def generate_overadpt_lazy_function():
    for api_name in OVERADPT_API_LIST:
        globals()[api_name] = create_lazy_overadpt_function(api_name)


def _patch_overadpt_ops():
    for api_name in OVERADPT_API_LIST:
        setattr(OVERADPT_MODULE, api_name, getattr(torch.ops.overadpt, api_name))


def _patch_overadpt_and_loadso():
    for api_name in OVERADPT_API_LIST:
        func = globals().get(api_name)
        setattr(OVERADPT_MODULE, api_name, func)


TORCH_NPU_USE_COMPATIBLE_IMPL = os.getenv("TORCH_NPU_USE_COMPATIBLE_IMPL")

if not (TORCH_NPU_USE_COMPATIBLE_IMPL == "1"):
    _add_overadpt_module()

    try:
        npu_path = pathlib.Path(__file__).parents[2]
        overadpt_so_path = os.path.join(npu_path, 'lib', 'libop_plugin_overadpt.so')
        from torch_npu.utils._path_manager import PathManager

        PathManager.check_directory_path_readable(overadpt_so_path)
        torch.ops.load_library(overadpt_so_path)
        import torch_npu.op_plugin.aclnn._aclnn_meta_registrations
    except OSError as e:
        nnal_strerror = ""
        if "undefined symbol" in str(e):
            nnal_strerror = "Please check the version of the NNAL package. " \
                            "An undefined symbol was found, " \
                            "which may be caused by a version mismatch between NNAL and torch_npu."
        NNAL_EX = OSError(e.errno, nnal_strerror)
        NNAL_EX.__traceback__ = e.__traceback__
        GLOBAL_E = e

    generate_overadpt_lazy_function()

