import torch
from torch.library import custom_op
from torch._inductor.codegen.common import IndentedBuffer
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.tools_common import legalize_graph
from torch.fx.passes.utils.fuser_utils import topo_sort, fuse_as_graphmodule, erase_nodes
from .dvm_build import TestOpSupport, DvmCodegenInterpreter

_dtype_map = {
    "float" : torch.float,
    "float32" : torch.float,
    "int" : torch.int,
    "int8" : torch.int8,
    "int16" : torch.int16,
    "int32" : torch.int,
    "long" : torch.long,
    "long int" : torch.long,
    "bfloat16" : torch.bfloat16,
    "float16" : torch.float16,
    "float64" : torch.double,
}

def _fused_run_stub(*args, **kwargs):
    assert(0)

def _fused_run_fake(*args, **kwargs):
    meta = args[-1]
    out_metas = meta[meta.find(":") + 1:].split(";")
    outputs = []
    for out_meta in out_metas:
        dtype, shape = out_meta.split("[")
        assert(dtype in _dtype_map)
        shape = [int(dim) for dim in shape[:-1].split(",")]
        outputs.append(args[0].new_empty(shape, dtype = _dtype_map[dtype]))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)

class GraphFusionPartitioner(CapabilityBasedPartitioner):
    fused_op_map_ = {}
    fused_id_ = 0

    def generate_fuse_code(self, sub_gm, orig_inputs, orig_outputs, fused_name) -> (str, str, str):
        ib = IndentedBuffer()
        ib.splice(f'\n"""\n{sub_gm.print_readable(print_output=False)}\n"""')
        ib.splice(f"@dvm.kernel")
        ib.splice(f"def {fused_name}(k):")
        codegen = DvmCodegenInterpreter(sub_gm, ib, True)
        codegen.run()
        code = ib.getvalue()
        meta_data = []
        for node in orig_outputs:
            val = node.meta["val"]
            meta_data.append(f"{str(val.dtype)[6:]}{list(val.shape)}")
        meta = f'{fused_name}:{";".join(meta_data)}'
        input_schema = ", ".join([f"Tensor x{i}" fori in range(len(orig_inputs))])
        output_schema = "Tensor" if len(orig_outputs) == 1 else f'({"Tensor, " * (len(orig_outputs) - 1)}Tensor)'
        schema = f"({input_schema}, str code, str meta) -> {output_schema}"
        return schema, meta, code

    def partition_and_fuse(self) -> torch.fx.GraphModule:
        partitions = self.propose_partitions()
        partition_nodes = [partition.nodes for partition in partitions]
        for partition in partition_nodes:
            fused_name = f"dvm_fused_{self.fused_id_}"
            self.fused_id_ += 1
            sorted_nodes = topo_sort(list(partition))
            sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
                self.graph_module,
                sorted_nodes,
                fused_name,
                partition,
                always_return_tuple=True
            )
            schema, meta, code = self.generate_fuse_code(sub_gm, orig_inputs, orig_outputs, fused_name)
            if schema in self.fused_op_map_:
                custom = self.fused_op_map_[schema]
            else:
                custom = custom_op("dvm::graph_fuse", _fused_run_stub, mutates_args=(), device_types="npu", schema=schema)
                custom.register_fake(_fused_run_fake)
                self.fused_op_map_[schema] = custom
            args = list(orig_inputs)
            args.append(code)
            args.append(meta)
            new_node = self.graph_module.graph.call_function(custom._opoverload, tuple(args), None)
            if len(orig_outputs) == 1:
                orig_outputs[0].replace_all_uses_with(new_node)
                new_node.meta["val"] = orig_outputs[0].meta.get("val", None)
            else:
                for i, orig_output in enumerate(orig_outputs):
                    proxy_out = torch.fx.Proxy(new_node)[i].node  # type: ignore[index]
                    orig_output.replace_all_uses_with(proxy_out)
                new_node.meta["val"] = tuple(orig_output.meta.get("val", None) for orig_output in orig_outputs)
            erase_nodes(self.graph_module, sorted_nodes)
        # topological sort original gm with newly created sub_gm
        legalize_graph(self.graph_module)
        return self.graph_module

def dvm_graph_fusion(graph : torch.fx.Graph):
    gm = graph.owning_module
    dvm_support = TestOpSupport()
    fusion_part = GraphFusionPartitioner(gm, dvm_support, allows_single_node_partition=True)
    fusion_part.partition_and_fuse()

def _dvm_generate_fallback_kernel(self, fallback_kernel, args):
    if fallback_kernel.op_overload._name != "dvm::graph_fuse":
        self.generate_extern_kernel_alloc(fallback_kernel, args)
        return
    code = fallback_kernel.constant_args[0]
    meta = fallback_kernel.constant_args[1]
    args = args[:-2]
    kernel_name = meta[:meta.find(":")]
    self.header.splice(code)
    buf_name = fallback_kernel.get_name()
    self.writeline(f"{buf_name} = {kernel_name}({', '.join(args)})")
    self.add_import_once(f"from torch_npu.op_plugin import dvm")

def _dvm_graph_fusion_patch():
    from torch._inductor import config
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    PythonWrapperCodegen.generate_fallback_kernel = _dvm_generate_fallback_kernel
    config.post_grad_custom_post_pass = dvm_graph_fusion

_dvm_graph_fusion_patch()
