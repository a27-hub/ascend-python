# coding: utf-8
import argparse
import ast
import contextlib
import os
import queue
import re
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from openpyxl import load_workbook

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


TARGET_OP_NAME = "aclnnMatmulReduceScatter"
REQUIRED_COLUMNS = {
    "op_name",
    "case_name",
    "input_tensor_shape",
    "input_tensor_range",
    "input_tensor_dtype",
}
DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
QUANT_REPEAT = 100


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


@dataclass(frozen=True)
class RunnerConfig:
    case_file: Optional[str] = None
    sheet_name: Optional[str] = None
    case_filter: Optional[str] = None
    comm_mode: str = "aiv"
    world_size: int = 4
    repeat: int = 1
    profile: bool = False


RUNNER_CONFIG = RunnerConfig(case_file=os.getenv("MM_RS_CASE_FILE"))


def _parse_literal_cell(value, excel_row, column_name, case_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Excel row {excel_row}, case '{case_name}', column "
            f"'{column_name}' must be a non-empty string"
        )
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"Excel row {excel_row}, case '{case_name}', column "
            f"'{column_name}' cannot be parsed: {value!r}"
        ) from error


def _make_tensor_spec(
        input_name, shape_value, range_value, dtype_value, excel_row, case_name):
    if not isinstance(shape_value, (list, tuple)) or not shape_value:
        raise ValueError(
            f"Excel row {excel_row}, case '{case_name}', {input_name} shape "
            f"must be a non-empty list, got {shape_value!r}"
        )
    if any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in shape_value):
        raise ValueError(
            f"Excel row {excel_row}, case '{case_name}', {input_name} shape "
            f"contains an invalid dimension: {shape_value!r}"
        )

    if not isinstance(range_value, (list, tuple)) or len(range_value) != 2:
        raise ValueError(
            f"Excel row {excel_row}, case '{case_name}', {input_name} range "
            f"must contain exactly two numbers, got {range_value!r}"
        )
    low, high = range_value
    if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (low, high)):
        raise ValueError(
            f"Excel row {excel_row}, case '{case_name}', {input_name} range "
            f"must be numeric, got {range_value!r}"
        )
    if low > high:
        raise ValueError(
            f"Excel row {excel_row}, case '{case_name}', {input_name} range "
            f"lower bound is greater than upper bound: {range_value!r}"
        )

    dtype_name = str(dtype_value).strip().lower()
    if dtype_name not in DTYPE_MAP:
        raise ValueError(
            f"Excel row {excel_row}, case '{case_name}', {input_name} has "
            f"unsupported dtype {dtype_value!r}; supported values are "
            f"{sorted(DTYPE_MAP)}"
        )

    return TensorSpec(
        shape=tuple(shape_value),
        value_range=(float(low), float(high)),
        dtype_name=dtype_name,
    )


def load_xlsx_cases(path, sheet_name=None, case_filter=None):
    if not path:
        raise ValueError(
            "XLSX case path is required; pass --case-file or set "
            "MM_RS_CASE_FILE"
        )
    if os.path.splitext(path)[1].lower() != ".xlsx":
        raise ValueError(f"Only .xlsx case files are supported, got: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"XLSX case file does not exist: {path}")

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(
                    f"Worksheet '{sheet_name}' does not exist in {path}; "
                    f"available sheets: {workbook.sheetnames}"
                )
            worksheet = workbook[sheet_name]
        else:
            worksheet = workbook.active

        rows = worksheet.iter_rows(values_only=True)
        try:
            header_row = next(rows)
        except StopIteration as error:
            raise ValueError(f"Worksheet '{worksheet.title}' is empty") from error

        header_index = {
            str(name).strip(): index
            for index, name in enumerate(header_row)
            if name is not None and str(name).strip()
        }
        missing_columns = sorted(REQUIRED_COLUMNS - set(header_index))
        if missing_columns:
            raise ValueError(
                f"Worksheet '{worksheet.title}' is missing required columns: "
                f"{missing_columns}"
            )

        def get_cell(row, column_name):
            index = header_index[column_name]
            return row[index] if index < len(row) else None

        cases = []
        for excel_row, row in enumerate(rows, start=2):
            op_name = get_cell(row, "op_name")
            if op_name is None or not str(op_name).strip():
                continue
            if str(op_name).strip() != TARGET_OP_NAME:
                continue

            case_name_value = get_cell(row, "case_name")
            case_name = (
                str(case_name_value).strip()
                if case_name_value is not None else ""
            )
            if not case_name:
                raise ValueError(
                    f"Excel row {excel_row} has an empty case_name"
                )
            if case_filter and case_filter not in case_name:
                continue

            shapes = _parse_literal_cell(
                get_cell(row, "input_tensor_shape"),
                excel_row,
                "input_tensor_shape",
                case_name,
            )
            ranges = _parse_literal_cell(
                get_cell(row, "input_tensor_range"),
                excel_row,
                "input_tensor_range",
                case_name,
            )
            dtypes = _parse_literal_cell(
                get_cell(row, "input_tensor_dtype"),
                excel_row,
                "input_tensor_dtype",
                case_name,
            )

            if not isinstance(dtypes, (list, tuple)) or len(dtypes) not in (2, 3):
                raise ValueError(
                    f"Excel row {excel_row}, case '{case_name}', "
                    "input_tensor_dtype must contain two entries (x1, x2) or "
                    f"three entries (x1, x2, bias), got {dtypes!r}"
                )
            if not isinstance(shapes, (list, tuple)) or len(shapes) < len(dtypes):
                raise ValueError(
                    f"Excel row {excel_row}, case '{case_name}', "
                    f"input_tensor_shape has insufficient entries: {shapes!r}"
                )
            if not isinstance(ranges, (list, tuple)) or len(ranges) < len(dtypes):
                raise ValueError(
                    f"Excel row {excel_row}, case '{case_name}', "
                    f"input_tensor_range has insufficient entries: {ranges!r}"
                )

            x1_spec = _make_tensor_spec(
                "x1", shapes[0], ranges[0], dtypes[0], excel_row, case_name
            )
            x2_spec = _make_tensor_spec(
                "x2", shapes[1], ranges[1], dtypes[1], excel_row, case_name
            )
            bias_spec = None
            if len(dtypes) == 3:
                bias_spec = _make_tensor_spec(
                    "bias",
                    shapes[2],
                    ranges[2],
                    dtypes[2],
                    excel_row,
                    case_name,
                )

            cases.append(
                MatmulReduceScatterCase(
                    case_name=case_name,
                    x1=x1_spec,
                    x2=x2_spec,
                    bias=bias_spec,
                )
            )

        if not cases:
            filter_message = (
                f" matching case filter {case_filter!r}" if case_filter else ""
            )
            raise ValueError(
                f"No {TARGET_OP_NAME} cases{filter_message} were found in "
                f"worksheet '{worksheet.title}'"
            )
        return cases
    finally:
        workbook.close()


def generate_tensor(spec, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    low, high = spec.value_range
    tensor = torch.rand(
        spec.shape, dtype=torch.float32, generator=generator
    )
    tensor = tensor * (high - low) + low
    return tensor.to(DTYPE_MAP[spec.dtype_name])


def normalize_case_tensors(case, x1, x2, bias, world_size):
    if x1.dim() != 2:
        raise ValueError(
            f"Case '{case.case_name}': x1 must be rank 2, got "
            f"{tuple(x1.shape)}"
        )
    if x2.dim() != 2:
        raise ValueError(
            f"Case '{case.case_name}': x2 must be rank 2, got "
            f"{tuple(x2.shape)}"
        )
    if x1.dtype != x2.dtype:
        raise ValueError(
            f"Case '{case.case_name}': x1 dtype {x1.dtype} does not match "
            f"x2 dtype {x2.dtype}"
        )

    m, k = x1.shape
    if m % world_size != 0:
        raise ValueError(
            f"Case '{case.case_name}': x1 M={m} is not divisible by "
            f"world_size={world_size}"
        )

    if x2.shape[0] == k:
        normalized_x2 = x2
    elif x2.shape[1] == k:
        normalized_x2 = x2.transpose(0, 1).contiguous()
    else:
        raise ValueError(
            f"Case '{case.case_name}': x2 shape {tuple(x2.shape)} is neither "
            f"[K, N] nor [N, K] for K={k}"
        )

    n = normalized_x2.shape[1]
    if bias is not None:
        if bias.dim() != 1 or bias.shape[0] != n:
            raise ValueError(
                f"Case '{case.case_name}': bias shape must be ({n},), got "
                f"{tuple(bias.shape)}"
            )
        if bias.dtype != x1.dtype:
            raise ValueError(
                f"Case '{case.case_name}': bias dtype {bias.dtype} does not "
                f"match x1 dtype {x1.dtype}"
            )

    return x1, normalized_x2, bias


def parse_runner_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--case-file",
        default=os.getenv("MM_RS_CASE_FILE"),
        help="XLSX case file path inside the szg container",
    )
    parser.add_argument("--sheet", dest="sheet_name")
    parser.add_argument("--case-filter")
    parser.add_argument("--comm-mode", default="aiv")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    args, unittest_args = parser.parse_known_args(argv[1:])

    if args.world_size <= 0:
        parser.error("--world-size must be greater than zero")
    if args.repeat <= 0:
        parser.error("--repeat must be greater than zero")

    config = RunnerConfig(
        case_file=args.case_file,
        sheet_name=args.sheet_name,
        case_filter=args.case_filter,
        comm_mode=args.comm_mode,
        world_size=args.world_size,
        repeat=args.repeat,
        profile=args.profile,
    )
    return config, [argv[0], *unittest_args]


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "50000")
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl", world_size=world_size, rank=rank
    )


def _get_hcom_name(rank):
    group = dist.distributed_c10d._get_default_group()
    if torch.__version__ > "2.0":
        hcom_name = group._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(rank)
    else:
        hcom_name = group.get_hccl_comm_name(rank)
    return (
        hcom_name.decode("utf-8")
        if isinstance(hcom_name, bytes) else hcom_name
    )


def _profiler_context(enabled, case_name, rank):
    if not enabled:
        return contextlib.nullcontext(None)

    safe_case_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", case_name)
    output_dir = os.path.join(
        ".", "result", safe_case_name, f"rank_{rank}"
    )
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=1, repeat=1, skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            output_dir
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )


def _construct_reference(x1, x2, bias, rank, world_size, output_dtype):
    reference = torch.matmul(
        x1.to(torch.float32), x2.to(torch.float32)
    )
    if bias is not None:
        reference = reference + bias.to(torch.float32)
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    rows_per_rank = x1.shape[0] // world_size
    reference = reference.narrow(
        0, rank * rows_per_rank, rows_per_rank
    )
    return reference.to(output_dtype)


def _execute_case(case, case_index, rank, hcom_name, config):
    seed_base = case_index * 1000 + rank * 10
    x1 = generate_tensor(case.x1, seed_base + 1)
    x2 = generate_tensor(case.x2, seed_base + 2)
    bias = (
        generate_tensor(case.bias, seed_base + 3)
        if case.bias is not None else None
    )
    x1, x2, bias = normalize_case_tensors(
        case, x1, x2, bias, config.world_size
    )

    x1 = x1.npu()
    x2 = x2.npu()
    bias = bias.npu() if bias is not None else None

    print(
        f"[Rank {rank}] Case {case.case_name}: "
        f"x1={tuple(x1.shape)}({x1.dtype}), "
        f"x2={tuple(x2.shape)}({x2.dtype}), "
        f"bias={'Y' if bias is not None else 'N'}, "
        f"comm_mode={config.comm_mode}",
        flush=True,
    )

    for iteration in range(config.repeat):
        enable_profile = (
            config.profile and case_index == 0 and iteration == 0
        )
        with _profiler_context(
                enable_profile, case.case_name, rank) as profiler:
            start_time = time.time()
            output = torch_npu.npu_mm_reduce_scatter_base(
                x1,
                x2,
                hcom_name,
                config.world_size,
                reduce_op="sum",
                bias=bias,
                comm_turn=0,
                comm_mode=config.comm_mode,
            )
            torch.npu.synchronize()
            elapsed = time.time() - start_time
            if profiler is not None:
                profiler.step()

        if iteration == 0:
            expected = _construct_reference(
                x1,
                x2,
                bias,
                rank,
                config.world_size,
                output.dtype,
            )
            torch.testing.assert_close(
                output.float(),
                expected.float(),
                rtol=0.05,
                atol=0.05,
            )

        if (
                iteration == 0
                or (iteration + 1) % 10 == 0
                or iteration == config.repeat - 1):
            print(
                f"[Rank {rank}] Case {case.case_name} iter "
                f"{iteration + 1}/{config.repeat}: OK ({elapsed:.4f}s)",
                flush=True,
            )


def _run_xlsx_worker(rank, config, cases, result_queue):
    failures = []
    try:
        _init_dist_hccl(rank, config.world_size)
        hcom_name = _get_hcom_name(rank)
        for case_index, case in enumerate(cases):
            try:
                _execute_case(
                    case, case_index, rank, hcom_name, config
                )
            except Exception:
                failures.append(
                    f"Case '{case.case_name}' failed on rank {rank}:\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                if dist.is_initialized():
                    dist.barrier()
    except Exception:
        failures.append(
            f"Worker rank {rank} setup failed:\n{traceback.format_exc()}"
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        result_queue.put((rank, failures))


class TestMmReduceScatterBase(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        _init_dist_hccl(rank, world_size)
        return dist

    @classmethod
    def _test_npu_mm_reduce_scatter_quant(cls, rank, input_list):
        (
            x1,
            x2,
            world_size,
            comm_mode,
            x1_scale,
            x2_scale,
            output_dtype,
            dequant_type,
            is_nz,
            init_pg,
            c2p,
        ) = input_list
        pg = init_pg(rank, world_size)
        hcom_name = _get_hcom_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        x2_scale = x2_scale.npu()
        x1_scale = x1_scale.npu()
        if is_nz:
            x2 = torch_npu.npu_format_cast(
                x2, torch_npu.Format.FRACTAL_NZ
            )

        for iteration in range(QUANT_REPEAT):
            start_time = time.time()
            if dequant_type:
                out, _ = torch_npu.npu_quant_mm_reduce_scatter(
                    x1,
                    x2,
                    hcom_name,
                    world_size,
                    reduce_op="sum",
                    bias=None,
                    comm_turn=0,
                    x1_scale=x1_scale,
                    x2_scale=x2_scale,
                    y_dtype=output_dtype,
                    comm_mode=comm_mode,
                    group_sizes=[128, 128, 128],
                )
            else:
                out, _ = torch_npu.npu_quant_mm_reduce_scatter(
                    x1,
                    x2,
                    hcom_name,
                    world_size,
                    reduce_op="sum",
                    bias=None,
                    comm_turn=0,
                    x2_scale=x2_scale,
                    y_dtype=output_dtype,
                    comm_mode=comm_mode,
                    group_sizes=[128, 128, 128],
                )
            torch.npu.synchronize()
            elapsed = time.time() - start_time

            if iteration == 0:
                c2p.put((rank, out.to(torch.float32).cpu()))

            if (
                    iteration % 10 == 0
                    or iteration == QUANT_REPEAT - 1):
                print(
                    f"[Rank {rank}] quant iter "
                    f"{iteration + 1}/{QUANT_REPEAT}: "
                    f"OK ({elapsed:.4f}s)",
                    flush=True,
                )

        pg.barrier()

    def _test_multiprocess(self, function, init_pg, input_list):
        (
            expected_output_list,
            x1_list,
            x2_list,
            world_size,
            comm_mode,
            x1_scale_list,
            x2_scale_list,
            output_dtype,
            dequant_type,
            is_nz,
        ) = input_list
        context = mp.get_context("spawn")
        child_to_parent = context.Queue(world_size)
        processes = []

        for rank in range(world_size):
            api_input_list = [
                x1_list[rank],
                x2_list[rank],
                world_size,
                comm_mode,
                x1_scale_list[rank],
                x2_scale_list[rank],
                output_dtype,
                dequant_type,
                is_nz,
                init_pg,
                child_to_parent,
            ]
            process = context.Process(
                target=function, args=(rank, api_input_list)
            )
            process.start()
            processes.append(process)

        for _ in range(world_size):
            rank, output = child_to_parent.get()
            self.assertRtolEqual(
                output, expected_output_list[rank], 0.05, 0.05
            )

        for process in processes:
            process.join()
            self.assertEqual(
                process.exitcode,
                0,
                f"child process failed: exitcode={process.exitcode}",
            )

    def _construct_quant_excepted_result(
            self,
            x1_list,
            x2_list,
            world_size,
            dequant_type,
            x1_scale_list=None,
            x2_scale_list=None):
        block = 128
        output = None
        for rank in range(world_size):
            x1_fp8 = x1_list[rank].to(torch.float32)
            x2_fp8 = x2_list[rank].to(torch.float32)
            x1_scale = x1_scale_list[rank]
            x2_scale = x2_scale_list[rank]

            x1_scale_full = x1_scale.repeat_interleave(
                block, dim=0
            ).repeat_interleave(block, dim=1)
            x2_scale_full = x2_scale.repeat_interleave(
                block, dim=0
            ).repeat_interleave(block, dim=1)

            x1_fp32 = x1_fp8 * x1_scale_full
            x2_fp32 = x2_fp8 * x2_scale_full
            local_output = torch.matmul(x1_fp32, x2_fp32)
            output = (
                local_output
                if output is None else torch.add(output, local_output)
            )

        rows_per_rank = x1_list[0].shape[0] // world_size
        return [
            output[
                rows_per_rank * rank:rows_per_rank * (rank + 1), :
            ]
            for rank in range(world_size)
        ]

    @skipIfUnsupportMultiNPU(4)
    @SupportedDevices(["Ascend950"])
    def test_npu_mm_reduce_scatter_base(self):
        config = RUNNER_CONFIG
        cases = load_xlsx_cases(
            config.case_file,
            sheet_name=config.sheet_name,
            case_filter=config.case_filter,
        )
        print(
            f"Loaded {len(cases)} XLSX case(s), "
            f"world_size={config.world_size}, "
            f"comm_mode={config.comm_mode}, repeat={config.repeat}",
            flush=True,
        )

        context = mp.get_context("spawn")
        result_queue = context.Queue(config.world_size)
        processes = []
        for rank in range(config.world_size):
            process = context.Process(
                target=_run_xlsx_worker,
                args=(rank, config, cases, result_queue),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        worker_results = {}
        for _ in range(config.world_size):
            try:
                rank, failures = result_queue.get(timeout=1)
            except queue.Empty:
                break
            worker_results[rank] = failures

        failure_messages = []
        for rank, process in enumerate(processes):
            if process.exitcode != 0:
                failure_messages.append(
                    f"Rank {rank} process exited with code "
                    f"{process.exitcode}"
                )
            if rank not in worker_results:
                failure_messages.append(
                    f"Rank {rank} did not return a test result"
                )
            else:
                failure_messages.extend(worker_results[rank])

        if failure_messages:
            self.fail("\n".join(failure_messages))

    @skipIfUnsupportMultiNPU(4)
    @SupportedDevices(["Ascend950"])
    def test_npu_mm_reduce_scatter_quant(self):
        world_size = 4
        m, k, n = 128, 512, 256
        for output_dtype in [torch.float16, torch.bfloat16]:
            x1_list = []
            x2_list = []
            x1_scale_list = []
            x2_scale_list = []
            dequant_type = 1
            for _ in range(world_size):
                x1_fp32 = torch.rand(
                    (m, k), dtype=torch.float32
                ) * 20 - 10
                x1 = x1_fp32.to(torch.float8_e5m2)
                x2_fp32 = torch.rand(
                    (k, n), dtype=torch.float32
                ) * 20 - 10
                x2 = x2_fp32.to(torch.float8_e5m2)
                x1_list.append(x1)
                x2_list.append(x2)
                x1_scale_list.append(
                    torch.randn(
                        (m // 128, k // 128), dtype=torch.float32
                    )
                )
                x2_scale_list.append(
                    torch.randn(
                        (k // 128, n // 128), dtype=torch.float32
                    )
                )

            expected_output_list = self._construct_quant_excepted_result(
                x1_list,
                x2_list,
                world_size,
                dequant_type,
                x1_scale_list,
                x2_scale_list,
            )
            for is_nz in [False]:
                self._test_multiprocess(
                    TestMmReduceScatterBase._test_npu_mm_reduce_scatter_quant,
                    TestMmReduceScatterBase._init_dist_hccl,
                    [
                        expected_output_list,
                        x1_list,
                        x2_list,
                        world_size,
                        "aiv",
                        x1_scale_list,
                        x2_scale_list,
                        output_dtype,
                        dequant_type,
                        is_nz,
                    ],
                )


if __name__ == "__main__":
    RUNNER_CONFIG, unittest_argv = parse_runner_args(sys.argv)
    sys.argv = unittest_argv
    if len(sys.argv) == 1:
        sys.argv.append(
            "TestMmReduceScatterBase.test_npu_mm_reduce_scatter_base"
        )
    run_tests()
