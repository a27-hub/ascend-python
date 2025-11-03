import torch

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch._dynamo.backends.common import aot_autograd
from torch._decomp import get_decompositions
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import CSE
from torch_npu._C import Kernel, BinaryOpType, KernelType, UnaryOpType
from typing import Any, Optional

from torch.fx.node import Argument, Target
aten = torch.ops.aten


class DvmCompilerKernel:
    def __init__(self, kernel: Kernel, kernel_name):
        self.kernel = kernel
        self.kernel.codegen()
        self.kernel.set_name(kernel_name)

    def run(self, *args, **kwargs):
        input_size = self.kernel.input_size()
        self.kernel.run(args[:input_size], args[input_size:])


class DvmCodegenInterpreter(torch.fx.Interpreter):
    def __init__(self, gm: torch.fx.GraphModule, compile_wrapper: IndentedBuffer):
        super().__init__(gm)
        self.cse = CSE()
        self.code = compile_wrapper
        self.code.do_indent()
        self.code.splice("kernel = Kernel(KernelType.kStaticShape)")
        self.node = None  # 初始化node属性

    def run_node(self, n: torch.fx.Node) -> Argument:
        self.node = n
        expr = super().run_node(n)
        if isinstance(expr, tuple):
            return tuple(self.cse.generate(self.code, i) for i in expr)
        else:
            return self.cse.generate(self.code, expr)

    def run(
        self,
        *args,
        initial_env: Optional[dict[torch.fx.Node, Any]] = None,
        enable_io_processing: bool = True,
    ) -> Any:
        with V.set_kernel_handler(Kernel()):
            super().run()
        self.code.splice("return kernel")

    def placeholder(self, target: "Target", args: tuple[Argument], kwargs: dict[str, Any]) -> Any:
        meta = torch.fx.traceback.current_meta["val"]
        if not hasattr(meta, 'shape') or not hasattr(meta, 'dtype'):
            raise ValueError(
                f"Missing shape or dtype in meta for placeholder {target}")
        return f"kernel.load({meta.shape}, {meta.dtype})"

    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        args_str = ", ".join(map(str, args))
        meta = torch.fx.traceback.current_meta["val"]
        if target in [aten.add.Tensor, aten.add.Scalar]:
            return f"kernel.binary(BinaryOpType.kAdd, {args_str})"
        elif target == aten.sub.Tensor:
            return f"kernel.binary(BinaryOpType.kSub, {args_str})"
        elif target in [aten.mul.Tensor, aten.mul.Scalar]:
            return f"kernel.binary(BinaryOpType.kMul, {args_str})"
        elif target == aten.div.Tensor:
            return f"kernel.binary(BinaryOpType.kDiv, {args_str})"
        elif target == aten.le.Scalar:
            return f"kernel.binary(BinaryOpType.kLess, {args_str})"
        elif target == aten.reshape.default:
            return f"kernel.reshape({args_str})"
        elif target == aten.sqrt.default:
            return f"kernel.unary(UnaryOpType.kSqrt, {args_str})"
        elif target == aten.mm.default:
            return f"kernel.matmul({args_str}, False, False)"
        elif target == torch.ops.npu._npu_dtype_cast.default:
            return f"kernel.cast({args_str})"
        elif target == aten.relu.default:
            return f"kernel.binary(BinaryOpType.kMax, {args_str}, 0)"
        elif target == aten.expand.default:
            return f"kernel.broadcast({args_str}, {meta.dtype})"
        else:
            raise ValueError(f"Unsupported operation: {target}.")

    def output(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        print(self.node)
        out = super().output(target, args, kwargs)
        if isinstance(out, tuple):
            input_nodes = self.node.all_input_nodes
            if len(input_nodes) != len(out):
                raise ValueError(
                    f"Output tuple length {len(out)} doesn't match input nodes length {len(input_nodes)}")

            return tuple(
                f"kernel.store(kernel.cast({arg}, {node.meta['val'].dtype}))"
                for arg, node in zip(out, input_nodes)
            )
        else:
            input_nodes = self.node.all_input_nodes
            if not input_nodes:
                raise ValueError("No input nodes found for output")
            return f"kernel.store(kernel.cast({out}, {input_nodes[0].meta['val'].dtype}))"


class FxDvmInterpreter(torch.fx.Interpreter):
    def __init__(self, gm:  torch.fx.GraphModule, kernel):
        super().__init__(gm)
        self.kernel = kernel
        self.node: torch.fx.Node

    def run_node(self, n: torch.fx.Node) -> Argument:
        self.node = n
        return super().run_node(n)

    def placeholder(self, target: "Target", args: tuple[Argument], kwargs: dict[str, Any]) -> Any:
        meta = self.node.meta["val"]
        return self.kernel.load(meta.shape, meta.dtype)

    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        if target in [aten.add.Tensor, aten.add.Scalar]:
            return self.kernel.binary(BinaryOpType.kAdd, *args)
        elif target == aten.sub.Tensor:
            return self.kernel.binary(BinaryOpType.kSub, *args)
        elif target == aten.mul.Tensor:
            return self.kernel.binary(BinaryOpType.kMul, *args)
        elif target == aten.le.Scalar:
            return self.kernel.binary(BinaryOpType.kLess, *args)
        elif target == aten.mm.default:
            return self.kernel.matmul(*args, False, False)
        elif target == torch.ops.npu._npu_dtype_cast.default:
            return self.kernel.cast(*args)
        elif target == aten.expand.default:
            return self.kernel.broadcast(*args, self.node.meta['val'].dtype)

    def output(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        out = super().output(target, args, kwargs)
        if isinstance(out, tuple):
            for arg, node in zip(out, self.node.all_input_nodes):
                op = self.kernel.store(self.kernel.cast(
                    arg, node.meta['val'].dtype))
        else:
            op = self.kernel.store(self.kernel.cast(
                out, self.node.all_input_nodes[0].meta['val'].dtype))
        return out


class DvmModule(torch.nn.Module):
    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__()
        self.gm = gm
        self.is_fx_dynamic = False
        self.kernel = Kernel(KernelType.kStaticShape)
        self.dvm_fx = FxDvmInterpreter(gm, self.kernel)
        self.dvm_fx.run()
        if not self.is_fx_dynamic:
            self.kernel.codegen()
            output_nodes = list(self.gm.graph.nodes)[-1].all_input_nodes

            def get_output_info(node):
                val = node.meta['val']
                return (val.size(), val.dtype, val.device)
            self.outputs_info = [get_output_info(
                node) for node in output_nodes]

    def __call__(self, *inputs):
        if self.is_fx_dynamic:
            # Todo
            self.kernel.codegen()
        else:
            outputs = [torch.empty(size, dtype=dtype, device=device)
                       for size, dtype, device in self.outputs_info]
        self.kernel.run(inputs, outputs)
        return outputs[0] if len(outputs) == 1 else outputs


DVM_SUPPORT_TYPE = [
    torch.bfloat16, torch.float16, torch.bool, torch.float32, torch.int32]


class TestOpSupport(OperatorSupportBase):

    def check_input(self, node: torch.fx.Node):
        input_dtypes = []

        for input_node in node.all_input_nodes:
            if not isinstance(input_node, torch.fx.Node):
                continue
            if not input_node.meta['val'].is_contiguous():
                return False
            input_dtypes.append(input_node.meta['val'].dtype)

        if len(input_dtypes) > 0:
            first_dtype = input_dtypes[0]
            if first_dtype not in DVM_SUPPORT_TYPE:
                return False
            return all(dtype == first_dtype for dtype in input_dtypes)

        return True

    def is_node_supported(self, submodules, node):
        if node.op != "call_function":
            return False
        if node.target in [aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.add.Scalar, aten.sub.Scalar, aten.mul.Scalar, aten.expand.default]:
            return self.check_input(node)
        return False


class DvmBackend:
    def __init__(self):
        self._compiled_module_cache: dict[torch.fx.GraphModule, torch.fx.GraphModule] = {
        }
        self.dvm_operator_support = TestOpSupport()

    def compile(self, graph_module: torch.fx.GraphModule, input_tensors) -> torch.fx.GraphModule:

        if graph_module in self._compiled_module_cache:
            compiled_module = self._compiled_module_cache[graph_module]
        else:

            partitioner = CapabilityBasedPartitioner(
                graph_module,
                self.dvm_operator_support,
                allows_single_node_partition=True
            )

            compiled_module = partitioner.partition_and_fuse(prefix="dvm_")
            compiled_module.print_readable()
            self._replace_fused_modules(compiled_module)
            self._compiled_module_cache[graph_module] = compiled_module

        return compiled_module

    def _replace_fused_modules(self, compiled_module: torch.fx.GraphModule):

        for node in compiled_module.graph.nodes:
            if node.op == "call_module" and "dvm" in node.name:
                fused_submodule = compiled_module.get_submodule(node.target)
                compiled_module.delete_submodule(node.target)
                compiled_module.add_submodule(
                    node.target,
                    DvmModule(fused_submodule)
                )

    def __call__(self, graph_module: torch.fx.GraphModule, input_args, **kwargs):
        decomposition_table = get_decompositions([
            aten.relu
        ])

        return aot_autograd(
            fw_compiler=self.compile,
            bw_compiler=self.compile,
            decompositions=decomposition_table,
        )(graph_module, input_args)
