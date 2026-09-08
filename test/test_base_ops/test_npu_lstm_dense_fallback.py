import copy
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuLstmDenseFallback(TestCase):
    def test_dense_fallback(self):
        if "950" not in torch.npu.get_device_name(0):
            self.skipTest("Ascend950 fallback test")
        torch.npu.set_compile_mode(jit_compile=False)
        # First case reproduces the ATK configuration. Others isolate the
        # no-bias, multi-layer, reverse-direction and batch-first branches.
        for layers, bias, bidirectional, batch_first in [
            (4, False, True, False), (1, False, False, False),
            (5, True, False, False), (1, True, True, True),
        ]:
            with self.subTest(layers=layers, bias=bias,
                              bidirectional=bidirectional, batch_first=batch_first):
                torch.manual_seed(1)
                cpu = torch.nn.LSTM(4, 4, num_layers=layers, bias=bias,
                                    bidirectional=bidirectional, batch_first=batch_first)
                npu = copy.deepcopy(cpu).npu()
                shape = (22, 30, 4) if batch_first else (30, 22, 4)
                states = (layers * (2 if bidirectional else 1), 22, 4)
                xs = [torch.empty(shape).uniform_(1, 5).requires_grad_(),
                      torch.zeros(states, requires_grad=True),
                      torch.empty(states).uniform_(-1, 1).requires_grad_()]
                ns = [t.detach().npu().requires_grad_() for t in xs]
                yc, (hc, cc) = cpu(xs[0], (xs[1], xs[2]))
                with torch.autograd.profiler.profile() as prof:
                    yn, (hn, cn) = npu(ns[0], (ns[1], ns[2]))
                    weights = [torch.randn_like(t) for t in (yc, hc, cc)]
                    sum((t * w.npu()).sum() for t, w in zip((yn, hn, cn), weights)).backward()
                    torch.npu.synchronize()
                sum((t * w).sum() for t, w in zip((yc, hc, cc), weights)).backward()
                pairs = list(zip((yc, hc, cc), (yn, hn, cn)))
                pairs += [(a.grad, b.grad) for a, b in zip(xs, ns)]
                pairs += [(a.grad, b.grad) for a, b in zip(cpu.parameters(), npu.parameters())]
                for ref, actual in pairs:
                    self.assertIsNotNone(actual)
                    self.assertIsNotNone(ref)
                    torch.testing.assert_close(actual.cpu(), ref, rtol=1e-3, atol=1e-3)
                names = [event.name for event in prof.function_events]
                self.assertTrue(any("npu_lstm" in name for name in names))
                self.assertFalse(any("aclnnLSTM" in name or "DynamicRNN" in name for name in names))


if __name__ == "__main__":
    run_tests()
