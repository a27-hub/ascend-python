import copy
import sys
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


def _print_diff(name, cpu_val, npu_val, is_fp16=False):
    """Print side-by-side comparison of CPU (expected) vs NPU (actual) values."""
    diff = cpu_val.astype(np.float64) - npu_val.astype(np.float64)
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / (np.abs(cpu_val.astype(np.float64)) + 1e-8)
    max_abs = np.max(abs_diff)
    max_rel = np.max(rel_diff)
    # Find the position of the largest absolute error
    max_idx = np.unravel_index(np.argmax(abs_diff), diff.shape)
    cpu_at_max = cpu_val.flat[np.argmax(abs_diff)]
    npu_at_max = npu_val.flat[np.argmax(abs_diff)]
    ratio = np.abs(npu_val.astype(np.float64)).max() / (np.abs(cpu_val.astype(np.float64)).max() + 1e-8)
    print(f"[{name}] shape={cpu_val.shape} dtype={cpu_val.dtype}/{npu_val.dtype} "
          f"fp16={is_fp16} max_abs_err={max_abs:.6e} max_rel_err={max_rel:.6e} "
          f"cpu_range=[{cpu_val.min():.6e}, {cpu_val.max():.6e}] "
          f"npu_range=[{npu_val.min():.6e}, {npu_val.max():.6e}] "
          f"max_ratio={ratio:.4f} "
          f"@max_err({max_idx}): cpu={cpu_at_max:.8e} npu={npu_at_max:.8e}",
          file=sys.stderr, flush=True)
    # Print a few sample values for manual inspection
    flat_len = min(diff.size, 3)
    for i in range(flat_len):
        print(f"  [{i}] cpu={cpu_val.flat[i]:.8e} npu={npu_val.flat[i]:.8e} "
              f"diff={diff.flat[i]:.6e} rel={rel_diff.flat[i]:.6e}",
              file=sys.stderr, flush=True)


class TestGruBackward(TestCase):
    # Common backward test: build CPU/NPU GRU, run forward then backward,
    # and compare gradients of input and all weight/bias tensors.
    # item format: [input_dtype_shape, h0_dtype_shape, input_size, hidden_size,
    #               num_layers, bidirectional, bias, batch_first]
    def _run_backward_case(self, item, use_h_grad=False):
        cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                               bidirectional=item[5], bias=item[-2], batch_first=item[-1])
        npu_gru = copy.deepcopy(cpu_gru).npu()

        is_fp16 = (item[0][0] == np.float16)
        # CPU side uses fp32 as high-precision reference; NPU side uses the case dtype.
        input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
        cpu_input1 = torch.from_numpy(input1.astype(np.float32))
        cpu_input1.requires_grad_(True)
        npu_input1 = torch.from_numpy(input1).npu()
        npu_input1.requires_grad_(True)

        h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
        cpu_h0 = torch.from_numpy(h0.astype(np.float32))
        npu_h0 = torch.from_numpy(h0).npu()

        if is_fp16:
            npu_gru = npu_gru.half()
            npu_input1 = npu_input1.half()
            npu_h0 = npu_h0.half()

        # Dump forward inputs for offline debugging / standalone repro
        _dump_tag = f"fp16_{item[2]}x{item[3]}_L{item[4]}" if is_fp16 else f"fp32_{item[2]}x{item[3]}_L{item[4]}"
        _dump_tag += f"_bidir{item[5]}_bias{item[-2]}_bfirst{item[-1]}"
        if use_h_grad:
            _dump_tag += "_hgrad"
        # CPU ref data
        torch.save({
            "input": cpu_input1.detach().cpu(),
            "h0": cpu_h0,
            "state_dict": cpu_gru.state_dict(),
            "item": item,
            "is_fp16": is_fp16,
        }, f"gru_fwd_cpu_{_dump_tag}.pth")
        # NPU data (tensors already on CPU for portability)
        torch.save({
            "input": npu_input1.detach().cpu(),
            "h0": npu_h0.cpu(),
            "state_dict": {k: v.cpu() for k, v in npu_gru.state_dict().items()},
            "item": item,
            "is_fp16": is_fp16,
        }, f"gru_fwd_npu_{_dump_tag}.pth")
        print(f"[dump] saved gru_fwd_cpu_{_dump_tag}.pth  /  gru_fwd_npu_{_dump_tag}.pth",
              file=sys.stderr, flush=True)

        cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
        npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

        # Debug: compare forward outputs (y, h)
        _print_diff("forward y", cpu_output_y.detach().numpy(), npu_output_y.detach().cpu().numpy(), is_fp16)
        _print_diff("forward h", cpu_output_h.detach().numpy(), npu_output_h.detach().cpu().numpy(), is_fp16)

        cpu_input1.retain_grad()
        npu_input1.retain_grad()

        if use_h_grad:
            # Backprop from sum of output_y and output_h so grad_h is non-zero, covering the dh path
            cpu_loss = cpu_output_y.sum() + cpu_output_h.sum()
            npu_loss = npu_output_y.sum() + npu_output_h.sum()
            cpu_loss.backward()
            npu_loss.backward()
        else:
            cpu_output_y.backward(torch.ones_like(cpu_output_y))
            npu_output_y.backward(torch.ones_like(npu_output_y))

        # Compare input gradient
        cpu_dx = cpu_input1.grad
        npu_dx = npu_input1.grad
        if is_fp16:
            _print_diff("grad input (dx)", cpu_dx.numpy().astype(np.float16), npu_dx.cpu().numpy(), is_fp16)
            self.assertRtolEqual(cpu_dx.numpy().astype(np.float16), npu_dx.cpu().numpy())
        else:
            _print_diff("grad input (dx)", cpu_dx.numpy(), npu_dx.cpu().numpy(), is_fp16)
            self.assertRtolEqual(cpu_dx.numpy(), npu_dx.cpu().numpy(), prec=1.e-1)

        # Compare gradients of all weight/bias tensors (same coverage as the legacy case)
        for (name_cpu, param_cpu), (name_npu, param_npu) in \
                zip(cpu_gru.named_parameters(), npu_gru.named_parameters()):
            assert name_cpu == name_npu, f"Param name mismatch: {name_cpu} vs {name_npu}"
            cpu_grad = param_cpu.grad
            npu_grad = param_npu.grad
            if is_fp16:
                _print_diff(f"grad {name_cpu}", cpu_grad.numpy().astype(np.float16), npu_grad.cpu().numpy(), is_fp16)
                self.assertRtolEqual(cpu_grad.numpy().astype(np.float16), npu_grad.cpu().numpy())
            elif "bias" in name_cpu:
                # bias gradients accumulate heavily, relax precision (consistent with the legacy case)
                _print_diff(f"grad {name_cpu}", cpu_grad.numpy(), npu_grad.cpu().numpy(), is_fp16)
                self.assertRtolEqual(cpu_grad.numpy(), npu_grad.cpu().numpy(), prec=1.e1)
            else:
                _print_diff(f"grad {name_cpu}", cpu_grad.numpy(), npu_grad.cpu().numpy(), is_fp16)
                self.assertRtolEqual(cpu_grad.numpy(), npu_grad.cpu().numpy(), prec=1.e-1)

    def test_gru_backward_fp32(self):
        # Covers the aclnnGRUBackward main path; matrix covers single/multi-layer,
        # uni/bi-directional, with/without bias, and both batch_first modes
        shape_format = [
            # [input, h0, input_size, hidden_size, num_layers, bidirectional, bias, batch_first]
            # single layer, unidirectional, with bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, True, False],
            # TODO: aclnnGRUBackward currently only supports single-layer unidirectional
            # and batch_first=False. Cases below are temporarily disabled pending
            # confirmation from CANN operator owner. Re-enable once supported.
            # # single layer, bidirectional, with bias, batch_first=True
            # [[np.float32, (2, 3, 4)], [np.float32, (2, 2, 3)], 4, 3, 1, True, True, True],
            # # two layers, unidirectional, with bias, batch_first=False
            # [[np.float32, (3, 2, 4)], [np.float32, (2, 2, 3)], 4, 3, 2, False, True, False],
            # single layer, unidirectional, without bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, False, False],
            # # single layer, bidirectional, without bias, batch_first=True
            # [[np.float32, (2, 3, 4)], [np.float32, (2, 2, 3)], 4, 3, 1, True, False, True],
            # # two layers, bidirectional, with bias, batch_first=False
            # [[np.float32, (3, 2, 4)], [np.float32, (4, 2, 3)], 4, 3, 2, True, True, False],
        ]
        for item in shape_format:
            self._run_backward_case(item)

    def test_gru_backward_fp16(self):
        # Covers the aclnnGRUBackward fp16 precision path
        shape_format = [
            # [input, h0, input_size, hidden_size, num_layers, bidirectional, bias, batch_first]
            [[np.float16, (3, 2, 4)], [np.float16, (1, 2, 3)], 4, 3, 1, False, True, False],
            # TODO: aclnnGRUBackward currently only supports single-layer unidirectional
            # and batch_first=False. Cases below are temporarily disabled pending
            # confirmation from CANN operator owner. Re-enable once supported.
            # [[np.float16, (3, 2, 4)], [np.float16, (2, 2, 3)], 4, 3, 1, True, True, False],
            # [[np.float16, (2, 3, 4)], [np.float16, (2, 2, 3)], 4, 3, 2, True, True, True],
            [[np.float16, (3, 2, 4)], [np.float16, (1, 2, 3)], 4, 3, 1, False, False, False],
        ]
        for item in shape_format:
            self._run_backward_case(item)

    def test_gru_backward_h_grad(self):
        # Covers backward path for output_h (h_n), verifying grad from both output_y and output_h
        shape_format = [
            # [input, h0, input_size, hidden_size, num_layers, bidirectional, bias, batch_first]
            [[np.float32, (3, 2, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, True, False],
            # TODO: aclnnGRUBackward currently only supports single-layer unidirectional
            # and batch_first=False. Cases below are temporarily disabled pending
            # confirmation from CANN operator owner. Re-enable once supported.
            # [[np.float32, (2, 3, 4)], [np.float32, (2, 2, 3)], 4, 3, 1, True, True, True],
            # [[np.float32, (3, 2, 4)], [np.float32, (4, 2, 3)], 4, 3, 2, True, True, False],
        ]
        for item in shape_format:
            self._run_backward_case(item, use_h_grad=True)


if __name__ == "__main__":
    run_tests()
