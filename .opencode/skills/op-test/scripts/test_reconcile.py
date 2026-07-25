#!/usr/bin/env python3
"""reconcile.py 单元测试 — 覆盖 MISS / DELETED-OK / EXTRA / BAD-LOG 四分支。"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

# 确保可以 import reconcile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconcile


class TestCollectTestMethods(unittest.TestCase):
    def test_collect_pos_neg(self):
        code = """
import unittest

class TestFoo(unittest.TestCase):
    def test_basic_fp16(self): pass
    def test_basic_bf16(self): pass
    def test_neg_invalid_dtype(self): pass
    def helper_method(self): pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            pos, neg = reconcile.collect_test_methods(f.name)
        os.unlink(f.name)

        self.assertEqual(pos, {"test_basic_fp16", "test_basic_bf16"})
        self.assertEqual(neg, {"test_neg_invalid_dtype"})

    def test_no_negatives(self):
        code = """
class TestBar:
    def test_only_pos(self): pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            pos, neg = reconcile.collect_test_methods(f.name)
        os.unlink(f.name)

        self.assertEqual(pos, {"test_only_pos"})
        self.assertEqual(neg, set())

    def test_skips_non_test_classes(self):
        """非 Test 前缀的类中的 test_ 方法不应被收集。"""
        code = """
class TestFoo(unittest.TestCase):
    def test_valid(self): pass

class HelperMixin:
    def test_should_be_ignored(self): pass

class SomeOtherClass:
    def test_also_ignored(self): pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            pos, neg = reconcile.collect_test_methods(f.name)
        os.unlink(f.name)

        self.assertEqual(pos, {"test_valid"})
        self.assertEqual(neg, set())


class TestParseAddDeleteLog(unittest.TestCase):
    def test_empty(self):
        d, a, raw = reconcile.parse_add_delete_log("")
        self.assertEqual(d, {})
        self.assertEqual(a, {})
        self.assertEqual(raw, [])

    def test_standard_format(self):
        text = """### Finalization Add/Delete Log

Delete test_dst_type_f8e5: 平台不适用 fp8_e5m2
Delete test_dst_type_f8e4m3: 平台不适用 fp8_e4m3fn
Delete test_dst_type_hf8: 平台不适用 hifloat8
Add test_extra_case: step3 interaction expansion
"""
        d, a, raw = reconcile.parse_add_delete_log(text)
        self.assertIn("test_dst_type_f8e5", d)
        self.assertEqual(d["test_dst_type_f8e5"], "平台不适用 fp8_e5m2")
        self.assertIn("test_dst_type_hf8", d)
        self.assertIn("test_extra_case", a)
        self.assertEqual(len(d), 3)
        self.assertEqual(len(a), 1)

    def test_comment_lines_ignored(self):
        text = """### Finalization Add/Delete Log

> 格式: Delete <exact_method_name>: <reason>
<!-- 这是注释 -->
Delete test_x: reason here
"""
        d, a, raw = reconcile.parse_add_delete_log(text)
        self.assertIn("test_x", d)
        self.assertEqual(len(d), 1)


class TestReconcileScenarios(unittest.TestCase):
    """端到端场景测试: 构造 CHECKLIST + 测试文件, 调用 reconcile() 检验结果。"""

    def _make_checklist(self, pos_items, neg_items, log_lines):
        """构造 CHECKLIST.md 内容。"""
        lines = ["## UT Generation Info", "", "### Coverage Matrix", ""]
        lines.append("| Test Item | Covered Params | Data Type | Dimensions | Scenario |")
        lines.append("|-----------|----------------|-----------|------------|----------|")
        for item in pos_items:
            lines.append(f"| {item} | x | float16 | (32,256) | Basic |")
        lines.append("")
        if neg_items:
            lines.append("### Negative Coverage Matrix")
            lines.append("")
            lines.append("| Test Item | Covered Params | Data Type | Dimensions | Scenario | Violated Field | Violation Construction | Expected |")
            lines.append("|-----------|----------------|-----------|------------|----------|----------------|------------------------|----------|")
            for item in neg_items:
                lines.append(f"| {item} | x | float16 | (32,256) | Negative | dtype | invalid | error |")
            lines.append("")
        if log_lines is not None:
            lines.append("### Finalization Add/Delete Log")
            lines.append("")
            for ll in log_lines:
                lines.append(ll)
            lines.append("")
        return "\n".join(lines)

    def _make_test_file(self, pos_methods, neg_methods):
        """构造测试文件内容。"""
        lines = ["import unittest", "", "class TestFoo(unittest.TestCase):"]
        if not pos_methods and not neg_methods:
            lines.append("    pass")
        for m in pos_methods:
            lines.append(f"    def {m}(self): pass")
        for m in neg_methods:
            lines.append(f"    def {m}(self): pass")
        return "\n".join(lines)

    def _run(self, pos_items, neg_items, log_lines, pos_methods, neg_methods):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as cl:
            cl.write(self._make_checklist(pos_items, neg_items, log_lines))
            cl.flush()
            cl_path = cl.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(self._make_test_file(pos_methods, neg_methods))
            tf.flush()
            tf_path = tf.name
        try:
            exit_code, report = reconcile.reconcile(cl_path, tf_path)
            return exit_code, "\n".join(report)
        finally:
            os.unlink(cl_path)
            os.unlink(tf_path)

    # ── MISS 分支 ──

    def test_miss_positive(self):
        """矩阵有行、代码无方法、日志无 Delete → [MISS] 硬失败。"""
        exit_code, report = self._run(
            pos_items=["test_basic_fp16", "test_basic_bf16"],
            neg_items=[],
            log_lines=[],
            pos_methods=["test_basic_fp16"],  # 缺 test_basic_bf16
            neg_methods=[]
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("[MISS] test_basic_bf16", report)
        self.assertIn("二选一", report)

    def test_miss_negative(self):
        """负向矩阵有行、代码无方法、日志无 Delete → [MISS] 硬失败。"""
        exit_code, report = self._run(
            pos_items=["test_basic_fp16"],
            neg_items=["test_neg_invalid_dtype", "test_neg_invalid_rank"],
            log_lines=[],
            pos_methods=["test_basic_fp16"],
            neg_methods=["test_neg_invalid_dtype"]  # 缺 test_neg_invalid_rank
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("[MISS] test_neg_invalid_rank", report)

    # ── DELETED-OK 分支 ──

    def test_deleted_ok(self):
        """矩阵有行、代码无方法、日志有 Delete → [DELETED-OK] 放行。"""
        exit_code, report = self._run(
            pos_items=["test_basic_fp16", "test_dst_type_f8e5"],
            neg_items=[],
            log_lines=["Delete test_dst_type_f8e5: 平台不适用 fp8_e5m2"],
            pos_methods=["test_basic_fp16"],  # 缺 test_dst_type_f8e5 但有日志
            neg_methods=[]
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("[DELETED-OK] test_dst_type_f8e5", report)

    def test_deleted_ok_multiple(self):
        """多条 Delete 日志, 全部放行。"""
        exit_code, report = self._run(
            pos_items=["test_basic_fp16", "test_dst_type_f8e5", "test_dst_type_hf8"],
            neg_items=[],
            log_lines=[
                "Delete test_dst_type_f8e5: fp8 不支持",
                "Delete test_dst_type_hf8: hifloat8 不支持",
            ],
            pos_methods=["test_basic_fp16"],
            neg_methods=[]
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("[DELETED-OK] test_dst_type_f8e5", report)
        self.assertIn("[DELETED-OK] test_dst_type_hf8", report)

    # ── EXTRA 分支 ──

    def test_extra_warning(self):
        """代码有方法但矩阵无行且无 Add 日志 → [EXTRA] 警告(不阻断)。"""
        exit_code, report = self._run(
            pos_items=["test_basic_fp16"],
            neg_items=[],
            log_lines=[],
            pos_methods=["test_basic_fp16", "test_extra_feature"],
            neg_methods=[]
        )
        self.assertEqual(exit_code, 0)  # EXTRA 不阻断
        self.assertIn("[EXTRA] test_extra_feature", report)

    def test_extra_silenced_by_add_log(self):
        """代码有方法、矩阵无行、但有 Add 日志 → 不报 EXTRA。"""
        exit_code, report = self._run(
            pos_items=["test_basic_fp16"],
            neg_items=[],
            log_lines=["Add test_extra_feature: interaction expansion"],
            pos_methods=["test_basic_fp16", "test_extra_feature"],
            neg_methods=[]
        )
        self.assertEqual(exit_code, 0)
        self.assertNotIn("[EXTRA] test_extra_feature", report)

    # ── BAD-LOG 分支 ──

    def test_bad_log_delete_nonexistent(self):
        """Delete 日志指向不存在的矩阵行 → [BAD-LOG] 警告。"""
        exit_code, report = self._run(
            pos_items=["test_basic_fp16"],
            neg_items=[],
            log_lines=["Delete test_nonexistent: 不存在的项"],
            pos_methods=["test_basic_fp16"],
            neg_methods=[]
        )
        self.assertEqual(exit_code, 0)  # BAD-LOG 不阻断
        self.assertIn("[BAD-LOG]", report)
        self.assertIn("test_nonexistent", report)

    def test_bad_log_natural_language(self):
        """Delete 日志名非逐字方法名(自然语言缩写) → 首词不在矩阵 → [BAD-LOG] + 矩阵行仍报 [MISS]。"""
        exit_code, report = self._run(
            pos_items=["test_dst_type_f8e5", "test_dst_type_f8e4m3"],
            neg_items=[],
            log_lines=["Delete fp8_all: 平台专属不支持 fp8 全系列"],  # "fp8_all" 不是任何矩阵行
            pos_methods=[],
            neg_methods=[]
        )
        # fp8_all 不匹配任何矩阵行 → BAD-LOG; 两行 f8e5/f8e4m3 仍报 MISS
        self.assertEqual(exit_code, 1)  # 有 MISS
        self.assertIn("[BAD-LOG]", report)
        self.assertIn("[MISS]", report)

    # ── 全通过分支 ──

    def test_all_pass(self):
        """矩阵与代码完全一致 → 退出码 0。"""
        exit_code, report = self._run(
            pos_items=["test_basic_fp16", "test_basic_bf16"],
            neg_items=["test_neg_invalid_dtype"],
            log_lines=[],
            pos_methods=["test_basic_fp16", "test_basic_bf16"],
            neg_methods=["test_neg_invalid_dtype"]
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("全覆盖", report)

    # ── 混合场景 ──

    def test_mixed_miss_and_deleted_ok(self):
        """部分 MISS、部分 DELETED-OK 混合 → 退出码 1, 两者各自正确分类。"""
        exit_code, report = self._run(
            pos_items=["test_a", "test_b", "test_c", "test_d"],
            neg_items=["test_neg_x", "test_neg_y"],
            log_lines=[
                "Delete test_c: not applicable",
                "Delete test_neg_y: not applicable",
            ],
            pos_methods=["test_a", "test_b"],   # 缺 test_c(有日志), test_d(无日志)
            neg_methods=["test_neg_x"],          # 缺 test_neg_y(有日志)
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("[MISS] test_d", report)
        self.assertIn("[DELETED-OK] test_c", report)
        self.assertIn("[DELETED-OK] test_neg_y", report)
        self.assertNotIn("[MISS] test_c", report)
        self.assertNotIn("[MISS] test_neg_y", report)


if __name__ == "__main__":
    unittest.main()
