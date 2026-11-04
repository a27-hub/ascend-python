"""Execute the production Meta function on CPU PyTorch without torch_npu.

These tests exercise real symbolic shapes; they do not execute GE or an NPU op.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def load_meta():
    path = Path(__file__).resolve().parents[2] / 'op_plugin/python/meta/_meta_registrations.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'npu_all_gather_quant_mm_meta')
    fn.decorator_list = []
    ns = dict(torch=torch, torch_npu=SimpleNamespace(float4_e2m1fn_x2=296),
              TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP={5: torch.float16, 15: torch.bfloat16})
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns[fn.name]


META = load_meta()


class HiF4DynamicMetaTest(unittest.TestCase):
    def call(self, m, layout=(128, 192), gather=True, **overrides):
        n = layout[1] if layout[0] == 128 else layout[0]
        s2 = (4, n, 1) if layout[0] == 128 else (n, 4, 1)
        args = dict(x1_scale=torch.empty((m, 4, 1), dtype=torch.int32, device='meta'),
                    x2_scale=torch.empty(s2, dtype=torch.int32, device='meta'),
                    x1_dtype=298, x2_dtype=298, x1_scale_dtype=299, x2_scale_dtype=299,
                    y_dtype=5, gather_output=gather)
        args.update(overrides)
        return META(torch.empty((m, 128), dtype=torch.uint8, device='meta'),
                    torch.empty(layout, dtype=torch.uint8, device='meta'), 'group', 2, **args)

    def test_unbacked_m_layouts(self):
        env = ShapeEnv()
        m = env.create_unbacked_symint()
        for layout in ((128, 192), (192, 128), (128, 128)):
            for gather in (False, True):
                y, g, amax = self.call(m, layout, gather)
                self.assertEqual(y.shape[0].node.expr, 2 * m.node.expr)
                self.assertEqual(y.shape[1], 128 if layout == (128, 128) else 192)
                self.assertEqual(y.dtype, torch.float16)
                if gather:
                    self.assertEqual(g.shape[0].node.expr, 2 * m.node.expr)
                    self.assertEqual(g.shape[1], 128)
                else:
                    self.assertEqual(tuple(g.shape), (0,))
                self.assertEqual(tuple(amax.shape), (0,))

    def test_invalid_contracts(self):
        for options in (dict(x1_scale_dtype=293), dict(gather_index=1), dict(amax_output=True),
                        dict(y_dtype=6), dict(x1_scale=None),
                        dict(x1_scale=torch.empty((65, 4, 1), dtype=torch.int32, device='meta')),
                        dict(x2_scale=torch.empty((192, 4, 1), dtype=torch.int32, device='meta'))):
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                self.call(64, **options)

    def test_static_layouts_and_output_types(self):
        for layout in ((128, 192), (192, 128), (128, 128)):
            for gather in (False, True):
                for enum, dtype in ((5, torch.float16), (15, torch.bfloat16)):
                    y, g, _ = self.call(64, layout, gather, y_dtype=enum)
                    self.assertEqual(tuple(y.shape), (128, 128 if layout == (128, 128) else 192))
                    self.assertEqual(y.dtype, dtype)
                    self.assertEqual(tuple(g.shape), (128, 128) if gather else (0,))

    def test_dynamic_kn_rejected(self):
        k = ShapeEnv().create_unbacked_symint()
        a = torch.empty((64, k), dtype=torch.uint8, device='meta')
        b = torch.empty((128, 192), dtype=torch.uint8, device='meta')
        with self.assertRaisesRegex(RuntimeError, 'static positive K/N'):
            META(a, b, 'group', 2, x1_dtype=298, x2_dtype=298,
                 x1_scale_dtype=299, x2_scale_dtype=299, y_dtype=5)

    def test_dynamo_reuses_graph_for_m(self):
        # Compile the production Meta calculation itself: proves its Python
        # checks do not specialize M, independently of TorchAir/GE execution.
        compilations = []

        def backend(gm, inputs):
            compilations.append(gm)
            return gm.forward

        def fn(a, sa, b, sb):
            return META(a, b, 'group', 2, x1_scale=sa, x2_scale=sb,
                        x1_dtype=298, x2_dtype=298, x1_scale_dtype=299,
                        x2_scale_dtype=299, y_dtype=5)

        compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=None)
        b = torch.empty((128, 192), dtype=torch.uint8)
        sb = torch.empty((4, 192, 1), dtype=torch.int32)
        torch._dynamo.mark_static(b)
        torch._dynamo.mark_static(sb)
        for m in (64, 96, 128, 64):
            a = torch.empty((m, 128), dtype=torch.uint8)
            sa = torch.empty((m, 4, 1), dtype=torch.int32)
            torch._dynamo.mark_dynamic(a, 0)
            torch._dynamo.mark_dynamic(sa, 0)
            torch._dynamo.mark_static(a, 1)
            torch._dynamo.mark_static(sa, 1)
            torch._dynamo.mark_static(sa, 2)
            y, g, _ = compiled(a, sa, b, sb)
            self.assertEqual(tuple(y.shape), (2 * m, 192))
            self.assertEqual(tuple(g.shape), (2 * m, 128))
        self.assertEqual(len(compilations), 1)


if __name__ == '__main__':
    unittest.main()
