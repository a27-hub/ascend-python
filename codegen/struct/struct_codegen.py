
import re
import itertools
from collections import namedtuple, defaultdict
from typing import List, Dict, Sequence, Union, Tuple
import yaml
import copy
from dataclasses import dataclass

from torchgen.code_template import CodeTemplate
from codegen.gen import FileManager
from codegen.model import (FunctionSchema, BaseTy, SchemaKind, BaseType,
                            NativeFunction, NativeFunctionsGroup, OperatorName)
from codegen.utils import concatMap
from codegen.context import native_function_manager
from codegen.api.types import NativeSignature
from codegen.api import cpp


PYTORCH_VERSION = '2.1.0.post3'


ACLNN_FUNCTIONS_DEFINITION = CodeTemplate("""\
${return_type} ${func_name}(${args_str}) {
    ${do_compatibility}
    ${check_or_apply_tensor}
    EXEC_NPU_CMD(${aclnnargs});
    ${infer_name}
    return ${result};
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


@dataclass(frozen=True)
class ResInfo:
    name: str
    size: str
    dtype: str
    option: str
    infer_name: str = None

    @staticmethod
    def parse(results: Dict[str, Dict[str, str]], func:'Union[NativeFunction, NativeFunctionsGroup]') -> List['ResInfo']:
        f = func
        
        tensor_number = sum(map(lambda x: x.type.name == BaseTy.Tensor, f.func.returns))
        if len(results) != tensor_number:
            raise RuntimeError(f"The number of result info in yaml is {len(results)}."
                               "That does not match {f.func.name}'s returns tensor number {tensor_number}")
        
        arguments = f.func.arguments.flat_all
        input_tensors = [arg.name for arg in arguments 
                        if isinstance(arg.type, BaseType) and arg.type.name == BaseTy.Tensor]
        
        res_infos = []
        for name, info in results.items():
            size = info.pop('size', None)
            dtype = info.pop('dtype', None)
            if size is None:
                raise RuntimeError(f"The {name}'s size  is None")
            if dtype is None:
                raise RuntimeError(f"The {name}'s dtype  is None")

            if size in input_tensors:
                size_formula = f'{size}.sizes()'
            else:
                size_formula = size
            if dtype in input_tensors:
                dtype_formula = f'{dtype}.scalar_type()'
            else:
                dtype_formula = dtype

            infer_name = info.pop('name', None)

            res_infos.append(
                ResInfo(name=name, size=size_formula, dtype=dtype_formula, option=input_tensors[0], infer_name=infer_name)
            )

        return res_infos


def split_name_params(schema: str) -> Tuple[str, List[str]]:
    m = re.match(r"(\w+)(\.\w+)?\((.*)\)", schema)
    if m is None:
        raise RuntimeError(f"Unsupported function schema: {schema}")
    name, _, params = m.groups()
    return name, params.split(", ")


@dataclass(frozen=True)
class StructInfo:
    name: str
    aclnn_name: str
    func: 'NativeFunction'
    cmd_args: str
    return_args: str
    acl_op: bool
    results: List[ResInfo] = None

    @staticmethod
    def from_yaml(
        es: Sequence[Dict[str, object]],
        grouped_native_functions: Union[NativeFunction, NativeFunctionsGroup],
    ) -> "List[StructInfo]":
        '''
        Parse a StructInfo from a dictionary as directly parsed 
        from struct.yaml
        '''
        native_functions = concatMap(
            # We need to pull out the view_inplace ops too, since they might have their own derivative entries.
            lambda g: [g]
            if isinstance(g, NativeFunction)
            else list(g.functions()),
            grouped_native_functions,
        )

        functions_by_schema: Dict[str, NativeFunction] = {}
        functions_by_signature: Dict[
            FunctionSchema, List[NativeFunction]
        ] = defaultdict(list)

        for function in native_functions:
            functions_by_signature[function.func.signature()].append(function)
            if str(function.func) in functions_by_schema:
                raise RuntimeError(f"{function.func} has multiple definitions")
            functions_by_schema[str(function.func)] = function
        
        functions_map: Dict[str, Dict] = {}
        for e in es:
            schema_str = e.get('func', None)
            op_name = schema_str.split('(')[0]
            functions_map[op_name] = copy.deepcopy(e)
            print(e)

        struct_infos: List[StructInfo] = []
        for e in es:
            schema_str = e.pop('func', None)
            structured_delegate = e.pop('structured_delegate', None)
            if structured_delegate is not None:
                e = functions_map[structured_delegate]
                print(e)
                e.pop('func')

            defn_name, _ = split_name_params(schema_str)
            print(defn_name)
            schema_function = functions_by_schema.get(schema_str)
            if not schema_function:
                avail = "\n".join(
                    k for k, v in functions_by_schema.items() if cpp.name(v.func) == defn_name
                )
                raise RuntimeError(
                    f"could not find ATen function for schema: {schema_str} "
                    f".  Available signatures:\n{avail}"
                )

            if 'op_api' not in schema_function.impl_ns:
                raise RuntimeError(f"The Aten function {schema_str} has no op_opi"
                                "in the impl_ns field of op_plugin_functions yaml")
            acl_op = 'acl_op' in schema_function.impl_ns
    
            aclnn_arguments = e.pop('exec', None)

            for key in e.keys():
                if not key.startswith('result'):
                    raise ValueError(f"Result infomations contains invalid key: {key}")

            aclnn_arguments = [argument.strip() for argument in aclnn_arguments.split(',')]
            aclnn_name = aclnn_arguments[0]

            if len(aclnn_arguments) == 1:
                with native_function_manager(schema_function):
                    sig = NativeSignature(schema_function.func, prefix='', symint=False)
                    for a in sig.arguments():
                        aclnn_arguments.append(a.name)

            aclnn_arguments = ', '.join(aclnn_arguments)   

            results = ResInfo.parse(e, schema_function)

            func_kind = schema_function.func.kind()

            if func_kind == SchemaKind.inplace:
                return_argument = [
                    schema_function.func.arguments.self_arg.argument.name
                ]
            else:
                return_argument = [result.name for result in results]
            
            if len(results) == 1:
                return_args = return_argument[0]
            elif func_kind == SchemaKind.out:
                return_args = f"std::forward_as_tuple({', '.join(return_argument)})"
            else:
                move_args = ', '.join(f'std::move({arg})' for arg in return_argument)
                return_args = f'std::make_tuple({move_args})'
            struct_info = StructInfo(
                name=defn_name,
                aclnn_name=aclnn_name,
                cmd_args=aclnn_arguments,
                func=schema_function,
                results=results,
                return_args=return_args,
                acl_op=acl_op,
            )
            struct_infos.append(struct_info)

        return struct_infos


def filt_op_branch(struct_ops: Dict) -> Dict:
    filt_ops = []
    for struct in struct_ops:
        branch = struct.pop('branch', 'base')
        if branch == 'base' or branch == 'v2r1':
            filt_ops.append(struct)
    return filt_ops


def parse_struct_yaml(path, grouped_native_functions: Union[NativeFunction, NativeFunctionsGroup]) -> List[StructInfo]:
    with open(path, 'r') as struct_file:
        struct_ops = yaml.safe_load(struct_file)

    filt_ops = filt_op_branch(struct_ops)
    struct_infos = StructInfo.from_yaml(filt_ops, grouped_native_functions)
    
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

        res_infos = struct.results

        do_compatibility = DO_COMPATIBILITY.substitute(aclnnname=struct.aclnn_name,
                                                       func_name=name,
                                                       args_exprs_str=args_exprs_str) if struct.acl_op else ""

        arguments = f.func.arguments.flat_non_out
        tensor_arguments = ", ".join([arg.name for arg in arguments 
                            if isinstance(arg.type, BaseType) and arg.type.name == BaseTy.Tensor])

        if kind == SchemaKind.out:
            apply_tensor_list = list(concatMap(
                lambda res_info: [CHECK_TENSOR.substitute(input=tensor_arguments,
                                                        result_name=res_info.name,
                                                        size=res_info.size,
                                                        dtype=res_info.dtype)],
                res_infos))
            aclnnargs = struct.cmd_args
        elif kind == SchemaKind.inplace:
            apply_tensor_list = []
            aclnnargs = struct.cmd_args
        else:
            apply_tensor_list = list(concatMap(
                lambda res_info: [APPLY_TENSOR.substitute(result_name=res_info.name,
                                                        size=res_info.size,
                                                        dtype=f'{res_info.option}.optoins().dtype({res_info.dtype})')],
                res_infos))
            aclnnargs = f'{struct.cmd_args}, {struct.return_args}'
        
        infer_name_list = list(concatMap(
                lambda res_info: INFER_NAME.substitute(result_name=res_info.name,
                                                       infer_func=res_info.infer_name)
                                if res_info.infer_name else "",
                res_infos))
        infer_name = "".join(infer_name_list)
            
        apply_tensor = "".join(apply_tensor_list)

        return_type = cpp.returns_type(f.func.returns).cpp_type()

    return [ACLNN_FUNCTIONS_DEFINITION.substitute(
            return_type=return_type,
            func_name=name,
            args_str=args_str,
            check_or_apply_tensor=apply_tensor,
            do_compatibility=do_compatibility,
            aclnnargs=aclnnargs,
            result=struct.return_args,
            infer_name=infer_name)]


def gen_op_api(
    fm: FileManager,
    struct_functions: Sequence[StructInfo]
) -> None:

    fm.write_with_template(
    f'StructAclnn.cpp', f'StructAclnn.cpp', lambda:{
    'op_api_definition':list(concatMap(
        lambda f: compute_op_api_definition(f),
        struct_functions
        ))}
    )
