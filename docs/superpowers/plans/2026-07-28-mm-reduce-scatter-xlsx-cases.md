# MM Reduce Scatter XLSX Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `test_mm_reduce_scatter_base.py` from one fixed 100-iteration input into a reproducible XLSX-driven PTA test runner for multiple `x1`/`x2`/optional-bias cases.

**Architecture:** The parent process parses and validates the workbook into picklable dataclasses, then spawns one worker per rank. Workers initialize HCCL once, iterate the same case list, generate deterministic rank-specific tensors, normalize `x2` to `[K, N]`, invoke `torch_npu.npu_mm_reduce_scatter_base`, and validate the first iteration against an all-reduced reference result.

**Tech Stack:** Python 3.11, `openpyxl`, PyTorch, `torch_npu`, HCCL, `unittest`

## Global Constraints

- Read `.xlsx` only; do not add `.xls` support or `xlrd`.
- The workbook path is a Linux path visible inside container `szg`.
- Required columns are `op_name`, `case_name`, `input_tensor_shape`, `input_tensor_range`, and `input_tensor_dtype`.
- Parse Python-like cell values with `ast.literal_eval`; never use `eval`.
- Treat two dtype entries as `bias=None`; treat three dtype entries as `x1`, `x2`, and `bias`.
- Normalize `[N, K]` x2 inputs to contiguous `[K, N]`.
- Default values are `world_size=4`, `comm_mode="aiv"`, and `repeat=1`.
- Preserve the existing quantized test as an independent test path.
- Do not stage or overwrite unrelated user changes.

---

## File Structure

- Modify `test/test_custom_ops/test_mm_reduce_scatter_base.py`: XLSX parsing, dataclasses, CLI handling, deterministic tensor generation, single-HCCL-group worker, PTA invocation, reference validation, and optional profiling.
- Create `test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py`: CPU-only unit tests for workbook parsing, bias inference, shape normalization, tensor ranges, and CLI argument separation.

### Task 1: XLSX Case Model and Parser

**Files:**
- Create: `test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py`
- Modify: `test/test_custom_ops/test_mm_reduce_scatter_base.py`

**Interfaces:**
- Produces: `TensorSpec(shape, value_range, dtype_name)`.
- Produces: `MatmulReduceScatterCase(case_name, x1, x2, bias)`.
- Produces: `load_xlsx_cases(path, sheet_name=None, case_filter=None) -> list[MatmulReduceScatterCase]`.

- [ ] **Step 1: Write failing parser tests**

Add tests that build temporary workbooks with `openpyxl.Workbook`:

```python
class TestLoadXlsxCases(unittest.TestCase):
    def _save_workbook(self, rows):
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.close()
        workbook = Workbook()
        sheet = workbook.active
        sheet.append([
            "op_name", "case_name", "input_tensor_shape",
            "input_tensor_range", "input_tensor_dtype",
        ])
        for row in rows:
            sheet.append(row)
        workbook.save(tmp.name)
        self.addCleanup(lambda: os.unlink(tmp.name))
        return tmp.name

    def test_three_dtypes_create_bias(self):
        path = self._save_workbook([[
            "aclnnMatmulReduceScatter", "with_bias",
            "[[8, 4], [4, 6], [6]]",
            "[(-1, 1), (-2, 2), (-3, 3)]",
            "['fp16', 'fp16', 'fp16']",
        ]])
        case = target.load_xlsx_cases(path)[0]
        self.assertEqual(case.x1.shape, (8, 4))
        self.assertEqual(case.x2.shape, (4, 6))
        self.assertEqual(case.bias.shape, (6,))

    def test_two_dtypes_ignore_third_shape_and_range(self):
        path = self._save_workbook([[
            "aclnnMatmulReduceScatter", "without_bias",
            "[[8, 4], [6, 4], [6]]",
            "[(-1, 1), (-2, 2), (-3, 3)]",
            "['bf16', 'bf16']",
        ]])
        case = target.load_xlsx_cases(path)[0]
        self.assertIsNone(case.bias)

    def test_filter_and_required_header_validation(self):
        path = self._save_workbook([[
            "aclnnMatmulReduceScatter", "selected_ID002",
            "[[8, 4], [4, 6], [6]]",
            "[(-1, 1), (-2, 2), (-3, 3)]",
            "['fp16', 'fp16', 'fp16']",
        ]])
        self.assertEqual(
            [case.case_name for case in target.load_xlsx_cases(path, case_filter="ID002")],
            ["selected_ID002"],
        )
```

Also test invalid extension, missing header, malformed literal, unsupported dtype, invalid list lengths, nonexistent sheet, and a filter that selects no rows. Assert that error messages contain the Excel row and `case_name` where available.

- [ ] **Step 2: Run tests and confirm the RED state**

Run:

```bash
cd /home/sangzhenguo/code/op-plugin
/root/anaconda3/envs/szg_py311/bin/python -m unittest \
  test.test_custom_ops.test_mm_reduce_scatter_xlsx_parser.TestLoadXlsxCases -v
```

Expected: FAIL because `load_xlsx_cases`, `TensorSpec`, and `MatmulReduceScatterCase` do not exist; the current target may first fail with the known profiling indentation error.

- [ ] **Step 3: Implement the parser**

Add immutable dataclasses and constants:

```python
REQUIRED_COLUMNS = {
    "op_name", "case_name", "input_tensor_shape",
    "input_tensor_range", "input_tensor_dtype",
}
SUPPORTED_DTYPES = {"fp16", "float16", "bf16", "bfloat16"}

@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    value_range: tuple[float, float]
    dtype_name: str

@dataclass(frozen=True)
class MatmulReduceScatterCase:
    case_name: str
    x1: TensorSpec
    x2: TensorSpec
    bias: Optional[TensorSpec]
```

Implement `_literal_cell`, `_tensor_spec`, and `load_xlsx_cases` using
`openpyxl.load_workbook(path, read_only=True, data_only=True)`. Use a header
name-to-index map, select only `aclnnMatmulReduceScatter`, infer bias from
dtype-list length, validate every selected row, close the workbook in
`finally`, and raise `ValueError` when no case remains.

- [ ] **Step 4: Run parser tests and confirm GREEN**

Run the Step 2 command.

Expected: all `TestLoadXlsxCases` tests PASS without initializing an NPU.

- [ ] **Step 5: Commit parser changes**

```bash
git add test/test_custom_ops/test_mm_reduce_scatter_base.py \
        test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py
git commit -m "test: parse mm reduce scatter XLSX cases"
```

### Task 2: Tensor Generation and Matrix Validation

**Files:**
- Modify: `test/test_custom_ops/test_mm_reduce_scatter_base.py`
- Modify: `test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py`

**Interfaces:**
- Consumes: `TensorSpec` and `MatmulReduceScatterCase`.
- Produces: `generate_tensor(spec, seed) -> torch.Tensor`.
- Produces: `normalize_case_tensors(case, x1, x2, bias, world_size) -> tuple[Tensor, Tensor, Optional[Tensor]]`.

- [ ] **Step 1: Write failing tensor/shape tests**

Add CPU-only tests:

```python
def test_generate_tensor_is_reproducible_and_in_range(self):
    spec = target.TensorSpec((16, 8), (-0.01, -0.001), "bf16")
    first = target.generate_tensor(spec, seed=123)
    second = target.generate_tensor(spec, seed=123)
    self.assertTrue(torch.equal(first, second))
    self.assertEqual(first.dtype, torch.bfloat16)
    self.assertGreaterEqual(first.float().min().item(), -0.011)
    self.assertLessEqual(first.float().max().item(), 0.0)

def test_normalize_transposes_nk_x2(self):
    case = make_case(x1_shape=(8, 4), x2_shape=(6, 4), bias_shape=(6,))
    x1 = torch.zeros(8, 4)
    x2 = torch.zeros(6, 4)
    bias = torch.zeros(6)
    _, normalized, _ = target.normalize_case_tensors(case, x1, x2, bias, 4)
    self.assertEqual(tuple(normalized.shape), (4, 6))
    self.assertTrue(normalized.is_contiguous())
```

Add rejection tests for x1 not rank-2, x2 not rank-2, incompatible K,
`M % world_size != 0`, and bias shape other than `(N,)`.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
cd /home/sangzhenguo/code/op-plugin
/root/anaconda3/envs/szg_py311/bin/python -m unittest \
  test.test_custom_ops.test_mm_reduce_scatter_xlsx_parser.TestTensorPreparation -v
```

Expected: FAIL because generation and normalization functions do not exist.

- [ ] **Step 3: Implement deterministic generation and validation**

Use a CPU generator:

```python
DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

def generate_tensor(spec, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    low, high = spec.value_range
    tensor = torch.rand(spec.shape, dtype=torch.float32, generator=generator)
    return (tensor * (high - low) + low).to(DTYPE_MAP[spec.dtype_name])
```

Implement normalization with explicit `case.case_name` in every exception.
Transpose only when `x2.shape[0] != K and x2.shape[1] == K`; otherwise reject
incompatible shapes. Validate optional bias after N is known.

- [ ] **Step 4: Run all CPU-only tests**

Run:

```bash
cd /home/sangzhenguo/code/op-plugin
/root/anaconda3/envs/szg_py311/bin/python -m unittest \
  test.test_custom_ops.test_mm_reduce_scatter_xlsx_parser -v
```

Expected: all parser and tensor tests PASS.

- [ ] **Step 5: Commit tensor preparation**

```bash
git add test/test_custom_ops/test_mm_reduce_scatter_base.py \
        test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py
git commit -m "test: generate validated mm reduce scatter inputs"
```

### Task 3: CLI Configuration and Single-Group Distributed Runner

**Files:**
- Modify: `test/test_custom_ops/test_mm_reduce_scatter_base.py`
- Modify: `test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py`

**Interfaces:**
- Produces: `RunnerConfig(case_file, sheet_name, case_filter, comm_mode, world_size, repeat, profile)`.
- Produces: `parse_runner_args(argv) -> tuple[RunnerConfig, list[str]]`.
- Produces: `_run_xlsx_worker(rank, config, cases, result_queue)`.

- [ ] **Step 1: Write failing CLI tests**

Test that custom arguments are consumed while unittest selectors remain:

```python
def test_parse_runner_args_preserves_unittest_args(self):
    config, remaining = target.parse_runner_args([
        "test_mm_reduce_scatter_base.py",
        "TestMmReduceScatterBase.test_npu_mm_reduce_scatter_base",
        "--case-file", "/tmp/cases.xlsx",
        "--comm-mode", "aiv",
        "--world-size", "4",
        "--repeat", "2",
        "--case-filter", "ID002",
    ])
    self.assertEqual(config.case_file, "/tmp/cases.xlsx")
    self.assertEqual(config.repeat, 2)
    self.assertEqual(
        remaining,
        [
            "test_mm_reduce_scatter_base.py",
            "TestMmReduceScatterBase.test_npu_mm_reduce_scatter_base",
        ],
    )
```

Test environment fallback `MM_RS_CASE_FILE`, positive world-size/repeat
validation, and default `aiv/4/1`.

- [ ] **Step 2: Run CLI tests and confirm RED**

Run:

```bash
cd /home/sangzhenguo/code/op-plugin
/root/anaconda3/envs/szg_py311/bin/python -m unittest \
  test.test_custom_ops.test_mm_reduce_scatter_xlsx_parser.TestRunnerArguments -v
```

Expected: FAIL because `RunnerConfig` and `parse_runner_args` do not exist.

- [ ] **Step 3: Implement CLI parsing**

Use `argparse.ArgumentParser(add_help=False)` and `parse_known_args`. Define
`--case-file`, `--sheet`, `--case-filter`, `--comm-mode`, `--world-size`,
`--repeat`, and `--profile`. Return a reconstructed argv containing the
program name followed by unknown/unittest arguments.

In `if __name__ == "__main__"`, assign the parsed configuration to a module
global, replace `sys.argv`, and when no unittest selector was supplied append
`TestMmReduceScatterBase.test_npu_mm_reduce_scatter_base` before `run_tests()`.

- [ ] **Step 4: Replace the base worker with one-process-group execution**

Implement `_run_xlsx_worker`:

```python
def _run_xlsx_worker(rank, config, cases, result_queue):
    pg = TestMmReduceScatterBase._init_dist_hccl(rank, config.world_size)
    try:
        hcom_name = _get_hcom_name(pg, rank)
        failures = []
        for case_index, case in enumerate(cases):
            x1 = generate_tensor(case.x1, case_index * 1000 + rank * 10 + 1)
            x2 = generate_tensor(case.x2, case_index * 1000 + rank * 10 + 2)
            bias = (
                generate_tensor(case.bias, case_index * 1000 + rank * 10 + 3)
                if case.bias is not None else None
            )
            x1, x2, bias = normalize_case_tensors(
                case, x1, x2, bias, config.world_size
            )
            failures.extend(
                _execute_case(case, x1, x2, bias, hcom_name, config, rank)
            )
            dist.barrier()
        result_queue.put((rank, failures))
    except Exception:
        result_queue.put((rank, [traceback.format_exc()]))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
```

`_execute_case` moves inputs to NPU and repeats the PTA call. On the first
iteration it computes local float32 matmul, adds bias before
`dist.all_reduce(SUM)`, slices `M/world_size` rows for the current rank,
converts to output dtype, and calls `torch.testing.assert_close` with
`rtol=0.05, atol=0.05`.

Update the base unittest method to load XLSX cases in the parent, spawn exactly
`world_size` workers once, collect one result per rank, join all processes,
assert every exit code is zero, and fail with an aggregated case summary.
Leave the quantized methods callable by their existing unittest name.

- [ ] **Step 5: Run CPU-only tests and syntax verification**

Run:

```bash
cd /home/sangzhenguo/code/op-plugin
/root/anaconda3/envs/szg_py311/bin/python -m unittest \
  test.test_custom_ops.test_mm_reduce_scatter_xlsx_parser -v
/root/anaconda3/envs/szg_py311/bin/python -m py_compile \
  test/test_custom_ops/test_mm_reduce_scatter_base.py \
  test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py
```

Expected: all tests PASS; `py_compile` exits 0. This also proves the existing
profiling indentation error has been removed.

- [ ] **Step 6: Commit CLI and runner**

```bash
git add test/test_custom_ops/test_mm_reduce_scatter_base.py \
        test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py
git commit -m "test: run XLSX mm reduce scatter cases"
```

### Task 4: Optional Profiler and NPU Verification

**Files:**
- Modify: `test/test_custom_ops/test_mm_reduce_scatter_base.py`
- Modify: `test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py`

**Interfaces:**
- Consumes: `RunnerConfig.profile`.
- Produces: `_profiler_context(enabled, case_name)` returning either a
  `torch_npu.profiler.profile` context or `contextlib.nullcontext`.

- [ ] **Step 1: Write failing profiler-selection test**

Mock `torch_npu.profiler.profile` and verify it is not constructed when
`enabled=False`; verify the output path includes a filesystem-safe case name
when `enabled=True`.

- [ ] **Step 2: Run profiler test and confirm RED**

Run:

```bash
cd /home/sangzhenguo/code/op-plugin
/root/anaconda3/envs/szg_py311/bin/python -m unittest \
  test.test_custom_ops.test_mm_reduce_scatter_xlsx_parser.TestProfilerSelection -v
```

Expected: FAIL because `_profiler_context` does not exist.

- [ ] **Step 3: Implement optional first-case profiling**

Move the existing profiler configuration into `_profiler_context`. Use
`contextlib.nullcontext(None)` when disabled. Enable it only for
`case_index == 0 and iteration == 0`, call `prof.step()` after the operator
invocation, and write to `result/<sanitized-case-name>`.

- [ ] **Step 4: Run the full CPU verification**

Run:

```bash
cd /home/sangzhenguo/code/op-plugin
/root/anaconda3/envs/szg_py311/bin/python -m unittest \
  test.test_custom_ops.test_mm_reduce_scatter_xlsx_parser -v
/root/anaconda3/envs/szg_py311/bin/python -m py_compile \
  test/test_custom_ops/test_mm_reduce_scatter_base.py \
  test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py
git diff --check
```

Expected: all unit tests PASS, compilation exits 0, and `git diff --check`
prints nothing.

- [ ] **Step 5: Create a small XLSX NPU smoke fixture**

Create `/tmp/mm_reduce_scatter_smoke.xlsx` with these four rows:

- fp16 with x2 `[K, N]` and bias.
- bf16 with x2 `[N, K]` and bias.
- bf16 with two dtype entries and therefore no bias.
- fp16 case excluded by `--case-filter`.

Use `M=128`, `K=512`, `N=256`, ranges `(-1, 1)`, and case names beginning
with `smoke_` for the three selected rows.

- [ ] **Step 6: Run the 4-card NPU smoke test**

Run:

```bash
cd /home/sangzhenguo/code/op-plugin
/root/anaconda3/envs/szg_py311/bin/python \
  test/test_custom_ops/test_mm_reduce_scatter_base.py \
  --case-file /tmp/mm_reduce_scatter_smoke.xlsx \
  --case-filter smoke_ \
  --world-size 4 \
  --comm-mode aiv \
  --repeat 1
```

Expected: three selected cases PASS on all four ranks, the unrelated row is
not loaded, and the final unittest result is `OK`.

- [ ] **Step 7: Verify representative workbook cases when the user file is available**

Run one fp16 bias case, one bf16 bias case, one transposed-x2 case, and one
two-dtype/no-bias case by changing `--case-filter`. Expected: each command
finishes with all four worker exit codes zero and precision PASS.

- [ ] **Step 8: Commit profiler and verification updates**

```bash
git add test/test_custom_ops/test_mm_reduce_scatter_base.py \
        test/test_custom_ops/test_mm_reduce_scatter_xlsx_parser.py
git commit -m "test: verify profiled XLSX reduce scatter cases"
```

