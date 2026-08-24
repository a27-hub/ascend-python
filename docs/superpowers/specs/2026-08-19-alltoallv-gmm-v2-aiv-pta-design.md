# AlltoAllv GMM V2 AIV PTA Design

## Goal

Expose the rewritten `aclnnAlltoAllvGroupedMatMulV2` AIV path through the official
`torch_npu.npu_alltoallv_gmm` PTA API, rebuild `torch_npu` from
`/home/sangzhenguo/code/op-plugin`, and validate it through the
`/home/sangzhenguo/code/ep_test` framework on Ascend 910C.

## Existing State

The installed `torch_npu 2.9.0.post2` schema does not contain `comm_mode`, so an
eager call with `comm_mode="aiv"` fails before dispatch. The current op-plugin
source already contains a compatibility-oriented optional `comm_mode` argument
and a V2 branch, but its generated package has not been installed in
`szg_py311`. The op-plugin API documentation and functional tests do not yet
prove the AIV V2 path.

## Selected Approach

Use the official op-plugin integration rather than a local PyBind extension or a
TorchAir-only workaround. The public schema remains backward compatible:

```text
npu_alltoallv_gmm(..., bool permute_out_flag=False,
                  str? comm_mode=None) -> (Tensor, Tensor, Tensor)
```

- `comm_mode=None` calls `aclnnAlltoAllvGroupedMatMul` (V1).
- Any explicitly supplied mode is forwarded to
  `aclnnAlltoAllvGroupedMatMulV2`; the V2 ACLNN implementation remains the
  authority for supported modes, devices, and dtypes.
- The test target uses `comm_mode="aiv"`, BF16, and Ascend 910C.
- The C++ bridge must materialize a null-terminated string before calling ACLNN.
- No changes are made to V1 behavior or to quantized sister operators.

## Components

### Op-plugin PTA layer

`op_plugin/config/op_plugin_functions.yaml` defines the optional schema field.
`op_plugin/ops/opapi/AllToAllvGmmNpuOpApi.cpp` allocates outputs and dispatches
V1 or V2 according to the optional mode. The meta implementation accepts the
same argument without changing output shapes. API documentation must describe
the compatibility behavior and the `aiv` example.

### Op-plugin tests

A focused regression test passes `comm_mode="aiv"` to the public PTA call and
checks BF16 GMM, optional shared-MM, and optional permute outputs against the
existing HCCL plus mathematical reference. A schema test confirms that the
installed operator accepts the keyword. The pre-rebuild installed package must
fail the schema probe, providing the RED phase.

### ep_test adapter

Register a new `a2avgmm_v2` operation in the existing CLI. Its data generator
creates deterministic per-rank BF16 inputs, expert-major/source-minor send and
receive counts, local expert weights, and optional shared-expert inputs. The
operation calls `torch_npu.npu_alltoallv_gmm` eagerly with `comm_mode="aiv"`.
The validator compares all requested outputs with an independent CPU FP32
reference and records the normal ep_test CSV result.

The first adapter scope is single-node EP2/EP4/EP8 on physical devices 8-15.
It covers transposed and non-transposed weights, zero-token experts, optional
shared MM, and optional permute output. Chunk strategies and non-BF16 dtypes are
out of scope.

## Error Handling

- Missing V2 ACLNN symbols produce the existing op-plugin CANN-version error.
- Unsupported `comm_mode`, dtype, or device is rejected by the V2 ACLNN API.
- Worker failures must propagate to a nonzero ep_test process exit code.
- Count-matrix inconsistencies are rejected during data generation before NPU
  execution.

## Verification

1. Run the schema probe against the old install and observe rejection of
   `comm_mode`.
2. Run op-plugin source/schema tests.
3. Build against PyTorch/TorchNPU 2.9.0 with Python 3.11 and install the wheel in
   `szg_py311`.
4. Confirm the installed dispatcher schema contains `str? comm_mode=None` and
   the binary references V2 ACLNN symbols.
5. Run a direct two-rank BF16 eager PTA smoke test with `comm_mode="aiv"`.
6. Run ep_test EP2, EP4, and EP8 accuracy cases and an invalid-mode case.
7. Re-run the ops-transformer operator build/unit suite if op definition defaults
   require adjustment.

