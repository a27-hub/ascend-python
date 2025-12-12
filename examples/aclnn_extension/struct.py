import os
import argparse
import yaml
import os
import stat
import functools
import hashlib
from typing import (List, Dict, Optional, Set, Callable, Any,
                    Union, TypeVar, Iterable, Tuple, Sequence)
from collections import defaultdict
import yaml
from torchgen.model import (BackendIndex, DispatchKey, NativeFunctionsGroup, OperatorName,
                            BackendMetadata, is_cuda_dispatch_key)
from torch_npu_gen.op_codegen.gen import FileManager
from torch_npu_gen.op_codegen.struct.struct_codegen import parse_struct_yaml, gen_op_api
from torch_npu_gen.op_codegen.struct.model import StructInfo
from torch_npu_gen.op_codegen.utils import PathManager, context
from torch_npu_gen.custom_functions import parse_custom_yaml
from torch_npu_gen.utils import rename_privateuse1_dispatch_key, get_torchgen_dir
from torch_npu_gen.op_codegen.model import NativeFunction


USED_KEYS = ['custom']


def parse_native_yaml_struct(
    native_yaml_path,
) -> List[NativeFunction]:
    with open(native_yaml_path, "r") as f:
        es = yaml.safe_load(f)

    rs: List[NativeFunction] = []
    if not es:
        return rs

    if 'custom' not in es:
        raise AssertionError("Can't find custom in yaml.")

    # if es['symint']:
    #     for e in es['symint']:
    #         global SYMINT_SET
    #         SYMINT_SET.add(e['func'].split("(")[0])

    all_funcs = []
    if es['custom']:
        all_funcs += es['custom']

    if not isinstance(all_funcs, list):
        raise TypeError("all_funcs must be a list")

    for e in all_funcs:
        funcs = e.get("func")
        with context(lambda: f"in:\n  {funcs}"):
            func, m = NativeFunction.from_yaml(e)
            rs.append(func)

    return rs


USED_KEYS = ['custom']


def filt_op_branch(struct_ops: Dict) -> Dict:
    support_ops = []
    for key in USED_KEYS:
        if struct_ops.get(key, None):
            support_ops += struct_ops[key]

    def filt_gen_opapi(op) -> bool:
        return 'gen_opapi' in op.keys()

    # for op in struct_ops['symint']:
    #     if is_support_version(op):
    #         SYMINT_OPS.add(op['func'].split("(")[0])

    filt_ops = list(filter(lambda op: filt_gen_opapi(op), support_ops))
    return filt_ops


def parse_struct_yaml(path, native_functions: Sequence[NativeFunction]) -> List[StructInfo]:
    path = os.path.realpath(path)
    PathManager.check_directory_path_readable(path)
    with open(path, 'r') as struct_file:
        struct_ops = yaml.safe_load(struct_file)

    filt_ops = filt_op_branch(struct_ops)
    struct_infos = StructInfo.from_yaml(filt_ops, native_functions, only_opapi=True)

    return struct_infos


def main() -> None:

    parser = argparse.ArgumentParser(description='Generate struct aclnn files')
    parser.add_argument(
        '-n',
        '--native_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '--struct_yaml',
        help='path to struct yaml file containing aclnn operators struct definitions')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    options = parser.parse_args()

    fm = FileManager(
            install_dir=options.output_dir, template_dir="codegen/struct/templates", dry_run=False
        )

    native_yaml_path = os.path.realpath(options.native_yaml)
    # torchgen_path = get_torchgen_dir()
    # tags_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/tags.yaml')
    native_functions = parse_native_yaml_struct(native_yaml_path)

    struct_info = parse_struct_yaml(options.struct_yaml, native_functions)
    gen_op_api(fm, struct_info)

if __name__ == '__main__':
    main()