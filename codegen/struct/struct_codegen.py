# Copyright (c) 2023 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
from collections import defaultdict
from typing import List, Dict, Sequence, Union, Tuple
import yaml
import copy
from dataclasses import dataclass

from torchgen.code_template import CodeTemplate
from codegen.gen import FileManager
from codegen.model import (FunctionSchema, BaseTy, SchemaKind, BaseType,
                           NativeFunction, NativeFunctionsGroup)
from codegen.utils import concatMap
from codegen.context import native_function_manager
from codegen.api.types import NativeSignature
from codegen.api import cpp
from .model import StructInfo


PYTORCH_VERSION = os.environ.get('PYTORCH_VERSION')[:3]
# PYTORCH_VERSION = '2.1.0.post3'[:3]

ACLNN_FUNCTIONS_DEFINITION = CodeTemplate("""\
${return_type} ${func_name}(${args_str}) {
    std::cout<<"in struct "<<"${func_name}"<<std::endl;
    ${do_compatibility}
    ${check_or_apply_tensor}
    EXEC_NPU_CMD(${aclnnargs});
    ${infer_name}
    return ${result};
}
""")

ACLNN_FUNCTIONS_DELEGATE = CodeTemplate("""\
${return_type} ${func_name}(${args_str}) {
    std::cout<<"in struct "<<"${func_name}"<<std::endl;
    return op_api::${func_name_out}(${args_str_out});
}
""")


DO_COMPATIBILITY = CodeTemplate("""\
DO_COMPATIBILITY(${aclnnname}, acl_op::${func_name}(${args_exprs_str}));
""")


APPLY_TENSOR = CodeTemplate("""\
at::Tensor ${result_name} = npu_preparation::apply_tensor_without_format(${size},
                            ${dtype});
""")


CHECK_TENSOR = CodeTemplate("""\
npu_preparation::check_tensor({${input}}, ${result_name}, ${dtype}, ${size});
""")


INFER_NAME = CodeTemplate("""\
at::namedinference::propagate_names(${result_name}, ${infer_func});
""")


def filt_op_branch(struct_ops: Dict) -> Dict:
    filt_ops = []
    for struct in struct_ops:
        branch = struct.pop('version', 'base')
        if branch == 'base' or branch == PYTORCH_VERSION:
            filt_ops.append(struct)
    return filt_ops


def parse_struct_yaml(path, native_functions: Sequence[NativeFunction]) -> List[StructInfo]:
    with open(path, 'r') as struct_file:
        struct_ops = yaml.safe_load(struct_file)

    filt_ops = filt_op_branch(struct_ops)
    struct_infos = StructInfo.from_yaml(filt_ops, native_functions)

    return struct_infos


def compute_op_api_definition(struct: StructInfo):
    f = struct.func

    with native_function_manager(f):
        kind = f.func.kind()
        sig = NativeSignature(f.func, prefix='', symint=False)
        name = cpp.name(f.func)
        args = sig.arguments()
        args_str = ', '.join(a.defn() for a in args)
        args_exprs_str = ', '.join(a.name for a in args)
        return_type = cpp.returns_type(f.func.returns).cpp_type()

        res_infos = struct.results

        if struct.structured_delegate is not None and kind == SchemaKind.inplace:
            delegate_function = struct.structured_delegate
            delegate_name = cpp.name(delegate_function.func)
            delegate_args_exprs_str = f'{args_exprs_str}, {f.func.arguments.self_arg.argument.name}'
            return [ACLNN_FUNCTIONS_DELEGATE.substitute(
                    return_type=return_type,
                    func_name=name,
                    args_str=args_str,
                    func_name_out=delegate_name,
                    args_str_out=delegate_args_exprs_str,)]

        do_compatibility = DO_COMPATIBILITY.substitute(aclnnname=struct.aclnn_name,
                                                       func_name=name,
                                                       args_exprs_str=args_exprs_str) if struct.acl_op else ""

        arguments = f.func.arguments.flat_non_out
        tensor_arguments = ", ".join([arg.name for arg in arguments
                                      if isinstance(arg.type, BaseType) and arg.type.name == BaseTy.Tensor])

        if kind == SchemaKind.out:
            apply_tensor_list = list(concatMap(
                lambda res_info:
                [CHECK_TENSOR.substitute(input=tensor_arguments,
                                         result_name=res_info.name,
                                         size=res_info.size,
                                         dtype=res_info.dtype)],
                res_infos))
        elif kind == SchemaKind.inplace:
            apply_tensor_list = []
        else:
            apply_tensor_list = list(concatMap(
                lambda res_info:
                [APPLY_TENSOR.substitute(result_name=res_info.name,
                                         size=res_info.size,
                                         dtype=f'{res_info.option}.options().dtype({res_info.dtype})')],
                res_infos))

        infer_name_list = list(concatMap(
            lambda res_info:
            INFER_NAME.substitute(result_name=res_info.name,
                                  infer_func=res_info.infer_name)
            if res_info.infer_name else "",
            res_infos))

        infer_name = "".join(infer_name_list)

        apply_tensor = "".join(apply_tensor_list)

    return [ACLNN_FUNCTIONS_DEFINITION.substitute(
            return_type=return_type,
            func_name=name,
            args_str=args_str,
            check_or_apply_tensor=apply_tensor,
            do_compatibility=do_compatibility,
            aclnnargs=struct.cmd_args,
            result=struct.return_args,
            infer_name=infer_name)]


def gen_op_api(
    fm: FileManager,
    struct_functions: Sequence[StructInfo]
) -> None:

    fm.write_with_template(
        f'StructKernelNpuOpapi.cpp', f'StructKernelNpuOpapi.cpp', lambda: {
            'op_api_definition': list(concatMap(
                lambda f: compute_op_api_definition(f),
                struct_functions
            ))}
    )
