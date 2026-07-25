"""parser 单测 — 验证 parse_def_cpp multi-config + parse_md_param_table 约束抽取。
用法: python test_constraint_info.py
"""
import sys, json, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dataclasses import asdict
from constraint_info import ConstraintInfo
from scaffold_init import parse_def_cpp, parse_md_param_table, GE_DTYPE_MAP

OPS_NN = Path("/home/c00618179/ops-nn")


class TestMultiConfigParse(unittest.TestCase):
    """验证 multi-config def 解析: platform_dtypes 非空 + 主块 dtype 正确"""

    def test_dynamic_quant_platforms(self):
        di = parse_def_cpp(str(OPS_NN / "quant/dynamic_quant/op_host/dynamic_quant_def.cpp"))
        self.assertIn("ascend950", di["platform_dtypes"],
                      "Bug1 regression: ascend950 must be in platform_dtypes")
        y950 = di["platform_dtypes"]["ascend950"].get("y", [])
        self.assertGreaterEqual(len(y950), 5,
                                f"ascend950 y should have ≥5 dtypes, got {len(y950)}")

    def test_dynamic_quant_main_block(self):
        di = parse_def_cpp(str(OPS_NN / "quant/dynamic_quant/op_host/dynamic_quant_def.cpp"))
        x_dtypes = [GE_DTYPE_MAP.get(d, d) for d in di["inputs"][0]["dtypes"]]
        self.assertIn("float16", x_dtypes)
        self.assertIn("bfloat16", x_dtypes)

    def test_group_quant_dtype_map(self):
        di = parse_def_cpp(str(OPS_NN / "quant/group_quant/op_host/group_quant_def.cpp"))
        gi = [p for p in di["inputs"] if p["name"] == "group_index"][0]
        gi_dtypes = [GE_DTYPE_MAP.get(d, d) for d in gi["dtypes"]]
        self.assertIn("int64", gi_dtypes, "DT_INT64 must map to int64")
        self.assertNotIn("DT_INT64", gi_dtypes, "Raw DT_INT64 must not appear")


class TestMdConstraintExtraction(unittest.TestCase):
    """验证 HTML md 约束抽取: shape_link/dtype_link/rank/dim_range"""

    @classmethod
    def setUpClass(cls):
        di = parse_def_cpp(str(OPS_NN / "quant/dynamic_quant/op_host/dynamic_quant_def.cpp"))
        cls._input = [{"name": i["name"],
                       "dtypes": [GE_DTYPE_MAP.get(d, d) for d in i["dtypes"]],
                       "param_type": i.get("param_type", "REQUIRED")}
                      for i in di["inputs"]]
        cls._output = [{"name": o["name"],
                        "dtypes": [GE_DTYPE_MAP.get(d, d) for d in o["dtypes"]]}
                       for o in di["outputs"]]
        cls.ci = parse_md_param_table(
            str(OPS_NN / "quant/dynamic_quant/docs/aclnnDynamicQuant.md"),
            cls._input, cls._output)

    def test_smooth_dtype_link(self):
        self.assertEqual(self.ci["smooth_scales"].dtype_link, "x")

    def test_smooth_shape_link(self):
        self.assertEqual(self.ci["smooth_scales"].shape_link,
                         {"ref": "x", "rule": "last_dim"})

    def test_smooth_optional(self):
        self.assertTrue(self.ci["smooth_scales"].optional,
                        "smooth_scales def ParamType=OPTIONAL")

    def test_y_shape_link_same(self):
        self.assertEqual(self.ci["y"].shape_link["rule"], "same",
                         "Bug2: y shape must be same (main condition)")

    def test_x_dim_range(self):
        self.assertEqual(self.ci["x"].dim_range, (2, 8))

    def test_scale_rank(self):
        self.assertIsNone(self.ci["scale"].rank)


class TestConstraintInfoSerialization(unittest.TestCase):
    """验证 dataclass 序列化/反序列化闭环"""

    def test_roundtrip(self):
        ci = ConstraintInfo(param="x", optional=False, dim_range=(2, 8),
                            valid_dtypes=["float16", "bfloat16"])
        d = asdict(ci)
        ci2 = ConstraintInfo(**d)
        self.assertEqual(ci2.param, "x")
        self.assertEqual(ci2.dim_range, (2, 8))

    def test_bad_field_raises(self):
        with self.assertRaises(TypeError):
            ConstraintInfo(**{"param": "x", "bad_field": 1})


class TestCoverageNegativeCases(unittest.TestCase):
    """验证 generate_negative_cases: mutex/large_shape/downgrade"""

    @classmethod
    def setUpClass(cls):
        di = parse_def_cpp(str(OPS_NN / "quant/dynamic_quant/op_host/dynamic_quant_def.cpp"))
        cls._input = [
            {"Param Name": i["name"], "Required/Optional": "Required" if i["param_type"] == "REQUIRED" else "Optional",
             "Type": "Tensor", "Supported dtypes": ", ".join(GE_DTYPE_MAP.get(d, d) for d in i["dtypes"])}
            for i in di["inputs"]]
        cls._input.append({"Param Name": "dst_type", "Required/Optional": "Optional", "Type": "int (Attr)", "Supported dtypes": "—"})
        cls._output = [
            {"Output Name": o["name"], "Type": "Tensor",
             "Dtype Rule": ", ".join(GE_DTYPE_MAP.get(d, d) for d in o["dtypes"])}
            for o in di["outputs"]]

        cls._input_params = [{"name": i["name"], "dtypes": [GE_DTYPE_MAP.get(d, d) for d in i["dtypes"]],
                             "param_type": i.get("param_type", "REQUIRED")}
                            for i in di["inputs"]]
        cls._output_specs = [{"name": o["name"], "dtypes": [GE_DTYPE_MAP.get(d, d) for d in o["dtypes"]]}
                             for o in di["outputs"]]
        cls.ci = parse_md_param_table(
            str(OPS_NN / "quant/dynamic_quant/docs/aclnnDynamicQuant.md"),
            cls._input_params, cls._output_specs)

    def test_mutex_empty_no_crash(self):
        """BUG-A 回归: mutex 空列表不动下标访问"""
        from scaffold_coverage import generate_negative_cases, CLASS_SHAPES
        rows = generate_negative_cases(
            self._input, self._output, list(self.ci.values()),
            "quantization", "float16", CLASS_SHAPES["quantization"], None)
        self.assertIsInstance(rows, list)

    def test_negative_categories_ge_4(self):
        """plan 验收: 负向类别 ≥4"""
        from scaffold_coverage import generate_negative_cases, CLASS_SHAPES
        rows = generate_negative_cases(
            self._input, self._output, list(self.ci.values()),
            "quantization", "float16", CLASS_SHAPES["quantization"], None)
        cats = set(r["Scenario"] for r in rows)
        self.assertGreaterEqual(len(cats), 4, f"negative categories={len(cats)}, scenarios={cats}")

    def test_invalid_dtype_not_in_valid_set(self):
        """dtype 非法值确实不在参数合法集合内(按违反字段匹配,不跨参数)"""
        from scaffold_coverage import generate_negative_cases, CLASS_SHAPES
        rows = generate_negative_cases(
            self._input, self._output, list(self.ci.values()),
            "quantization", "float16", CLASS_SHAPES["quantization"], None)
        # 构建参数→合法dtype映射
        valid_map = {}
        for ip in self._input:
            if ip["Supported dtypes"] != "—":
                valid_map[ip["Param Name"]] = set(ip["Supported dtypes"].split(", "))
        for o in self._output:
            if o.get("Dtype Rule", "—") != "—":
                valid_map[o["Output Name"]] = set(o["Dtype Rule"].split(", "))

        for r in rows:
            if "dtype=" in r["Violation Construction"]:
                invalid_dt = r["Violation Construction"].split("=")[1]
                violator = r["Violated Field"]
                if violator in valid_map:
                    self.assertNotIn(invalid_dt, valid_map[violator],
                                     f"{violator} invalid dtype={invalid_dt} should not be in valid set {valid_map[violator]}")


class TestCoverageBaselineNoConstraintParsing(unittest.TestCase):
    """B 侧砍除后回归: generate_coverage 只产基线矩阵, 不解析「约束」列,
    不删行、不建交互占位行(行增删交步骤3 模型)。"""

    def test_no_parse_apply_functions(self):
        """parse_constraints / apply_constraints / _split_top_level 已删除"""
        import scaffold_coverage as sc
        for fn in ("parse_constraints", "apply_constraints", "_split_top_level"):
            self.assertFalse(hasattr(sc, fn), f"{fn} 应已删除(B 侧 DSL 解析)")

    def test_baseline_no_placeholder_row(self):
        """约束列标 不适用/交互 不影响基线矩阵——脚本不读约束列, 无占位行、无删行"""
        from scaffold_coverage import generate_coverage, CLASS_SHAPES
        input_params = [
            {"Param Name": "x", "Type": "Tensor", "Required/Optional": "Required", "Supported dtypes": "float16, bfloat16", "Constraints": "≥2 dims"},
            {"Param Name": "smooth_scales", "Type": "Tensor", "Required/Optional": "Optional", "Supported dtypes": "float16, bfloat16",
             "Constraints": "Not applicable: smooth_scales, test use"},  # Constraints column says not applicable, script should ignore (not delete row)
        ]
        output_specs = [{"Output Name": "y", "Dtype Rule": "int8", "Shape Rule": "same as x"}]
        rows = generate_coverage(input_params, output_specs, "quantization", [], None)
        # No placeholder rows (rows with scenario containing "see constraints")
        placeholder = [r for r in rows if "see constraints" in r.get("Scenario", "")]
        self.assertEqual(placeholder, [], "脚本不应再建交互占位行")
        # smooth_scales 仍出现在 product 组合行(脚本未因'不适用'删行)
        has_smooth = any("smooth_scales" in r.get("Covered Params", "") for r in rows)
        self.assertTrue(has_smooth, "Script does not parse Constraints column; 'Not applicable' marker should not cause row deletion (deletion delegated to step 4)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
