#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_foreach_nc.py — #640 相邻成员矩阵探针：_foreach_ 族非连续/视图输入 CPU×NPU 语义对照
（真机调研 20261025 第三轮，§7.1；#640 = _foreach_add_ 转置视图静默错位写，op-plugin 已立案）

背景：#640 实锤形态——对 (4,3) 转置视图（stride=(1,4)）做 _foreach_add_，NPU 按 flat 偏移
      0/5/10（伪 stride=5）错位读写（期望 0/4/8），连续对照全对，三芯同形态。
      本探针回答两个未决问题：
      ① #640 是 _foreach_add_ 独有还是 _foreach_ 族级通病（决定修复 PR 范围：单 kernel vs 共享地址层）；
      ② 错位发生在读侧还是写侧（zero_ 纯写 / copy_ 写+读他 / add_ 读己+读他+写 / unary / norm 纯读 的诊断阶梯）。

矩阵：19 成员 × 5 输入形态 = 95 格 + c000 锚点案（#640 原貌无警戒区复刻）。
  成员（诊断阶梯）：zero_纯写 → copy_写+读他(源恒连续) → add_/#640原生 → add_/mul_/sub_/div_/clamp_max_/
    clamp_min_/maximum_ 标量族 → exp_/neg_/reciprocal_ 一元族 → lerp_/addcmul_/addcdiv_ 优化器三参族
    → add(非就地)/exp(非就地) → norm(纯读)。
  形态：contig 连续(对照锚) / transpose 转置视图(#640 原形, stride=(1,4)) / slice 步长切片
    (base[:, ::2], stride=(6,2)) / negstep 负步长(base.flip(1)，物化拷贝) / offset 行切片
    (base[1:]，连续步长+非零 storage_offset=3)。
  统一视图形状 (4,3)——与 #640 立案形状一致，跨格可比。

判定四通道（每格独立子进程，崩溃即证据）：
  ① 视图值 NPU vs CPU 逐位（exact）/容差（close，仅超越函数族）；
  ② 全存储逐位——基底带 8 元素**尾部警戒区**：伪步长越界写会落入警戒区（立案案伪 stride=5 在
     (4,3) 视图上会写出 flat 13/14，越过 12 元素精确存储=OOB；警戒区使其可见）；
  ③ 改写掩码 vs 视图覆盖掩码——"原存储是否被改写/是否越权写/是否静默不生效"的显式证据；
  ④ stride/offset/contig 元数据哨兵（NPU flip 产物若非连续即在此暴露）。
失败形态分类器（追加在 detail）：GUARD/OOW-WRITE（越权写警戒区或视图外=数据损坏加重证据）、
  NOOP-SILENT（静默不生效=上游 GPU 同构症状）、INVIEW-MISWRITE（区内错值/错序=#640 同形）、
  VAL-DIFF(mask同)（纯数值差=ULP 噪声，#621 族，不混入错位结论）、CHILD-CRASH。

铁律（对齐 probe_complex_sem.py v2）：seed 与全部输入构造固定在 make_cases 工厂作用域，build/run
  零 RNG 零构造（run 内只 .to(dev).clone()+视图+算子——clone 防 CPU 侧 .to("cpu") 返本体裁改工厂张量，
  selfcheck 哨兵据此把关）；CPU 参照先行；每案标注期望形状（expect 字段 + meta 哨兵张量）；
  子进程隔离（错位写若踩坏 allocator → CHILD-CRASH 定格在该格，不级联）。

用法：
  python3 probe_foreach_nc.py                  # 父进程：逐格子进程调度 + 比对 + 矩阵透视表
  python3 probe_foreach_nc.py __case__ 7       # 子进程模式（内部使用）
  python3 probe_foreach_nc.py __selfcheck__    # RNG 不变性哨兵：双跑不重置 seed，载荷必须逐位一致
产物：stdout 报告（含 19×5 透视表）+ /tmp/probe_foreach_nc_results.json
已知注意：① negstep 格的算子目标物是 flip 物化产物（自有存储），改写掩码双侧恒零属预期
  （PyTorch 稠密张量无负步长视图，as_strided 负步长报错），该格仅验证值语义 + flip 产物步长哨兵；
  ② maximum_ 在 torch2.7.1 可能缺签名 → ERR-SYMM=签名档非缺陷（看报错文案）；
  ③ 真机 96 子进程 × 约 8s ≈ 15 分钟级，建议 950 首跑、A2/A3 复证。
"""
import json
import math
import subprocess
import sys

DEVICE = "npu:0"
RESULT_PATH = "/tmp/probe_foreach_nc_results.json"

FORMS = ["contig", "transpose", "slice", "negstep", "offset"]

# 成员表：(mid, 诊断阶梯 tag, kind, 比对模式, rtol, atol, 备注)
MEMBERS = [
    ("zero",        "W0-纯写",            "inplace",  "exact", 0,    0,    "无读路径；错则写地址错"),
    ("copy",        "W1-写+读他(源连续)", "inplace",  "exact", 0,    0,    "源恒连续：错则写侧嫌疑"),
    ("add_list",    "RW-读己读他写",      "inplace",  "exact", 0,    0,    "#640 原生形态(双列表)"),
    ("add_scalar",  "RW-标量",            "inplace",  "exact", 0,    0,    "优化器 lr 路径形态"),
    ("mul_scalar",  "RW-标量",            "inplace",  "exact", 0,    0,    ""),
    ("sub_scalar",  "RW-标量",            "inplace",  "exact", 0,    0,    ""),
    ("div_scalar",  "RW-标量",            "inplace",  "exact", 0,    0,    ""),
    ("clamp_max",   "RW-标量",            "inplace",  "exact", 0,    0,    ""),
    ("clamp_min",   "RW-标量",            "inplace",  "exact", 0,    0,    ""),
    ("maximum",     "RW-标量",            "inplace",  "exact", 0,    0,    "2.7.1 签名可能缺→ERR-SYMM=签名档"),
    ("exp_",        "U-一元读己写",       "inplace",  "close", 1e-5, 1e-6, ""),
    ("neg_",        "U-一元",             "inplace",  "exact", 0,    0,    "位级安全（无舍入）"),
    ("reciprocal_", "U-一元",             "inplace",  "close", 1e-9, 1e-9, ""),
    ("lerp_",       "OPT-三参",           "inplace",  "close", 1e-5, 1e-6, "ULP 噪声已知(#621/C4)→close"),
    ("addcmul_",    "OPT-三列表",         "inplace",  "close", 1e-5, 1e-6, "fused 舍入→close"),
    ("addcdiv_",    "OPT-三列表",         "inplace",  "close", 1e-5, 1e-6, "fused 舍入→close"),
    ("add_oop",     "OO-非就地",          "ooplace",  "exact", 0,    0,    "非就地：读侧证据对照"),
    ("exp_oop",     "OO-非就地",          "ooplace",  "close", 1e-5, 1e-6, ""),
    ("norm",        "RO-纯读",            "readonly", "close", 1e-5, 1e-5, "无写路径；错则读地址错"),
]

GUARD = 8          # 尾部警戒区元素数
V0, V1 = 4, 3      # 统一视图形状（#640 立案形状）
# 各形态的基底形状（view=4×3 由形态变换产生）
BASE_SHAPE = {"contig": (4, 3), "transpose": (3, 4), "slice": (4, 6),
              "negstep": (4, 3), "offset": (5, 3)}
# 各形态期望视图形状 / 步长 / 偏移（expect 标注 + 与设备端实测 meta 哨兵互证）
VIEW_EXPECT = {
    "contig":    {"view": [4, 3], "stride": [3, 1], "offset": 0, "contig": 1},
    "transpose": {"view": [4, 3], "stride": [1, 4], "offset": 0, "contig": 0},
    "slice":     {"view": [4, 3], "stride": [6, 2], "offset": 0, "contig": 0},
    "negstep":   {"view": [4, 3], "stride": [3, 1], "offset": 0, "contig": 1},
    "offset":    {"view": [4, 3], "stride": [3, 1], "offset": 3, "contig": 1},
}


def apply_form(base, form):
    if form == "contig":
        return base                      # (4,3) 连续
    if form == "transpose":
        return base.t()                  # (3,4)→(4,3) stride(1,4) —— #640 原形
    if form == "slice":
        return base[:, ::2]              # (4,6)→(4,3) stride(6,2)
    if form == "negstep":
        return base.flip(1)              # (4,3) 物化负步长拷贝（自有存储，连续）
    if form == "offset":
        return base[1:]                  # (5,3)→(4,3) offset=3 stride(3,1) 连续
    raise ValueError(form)


# ---------------- 案例定义（工厂作用域：全部输入一次性构造，seed 固定） ----------------
def make_cases():
    import torch
    torch.manual_seed(640)
    cases = []

    # 视图覆盖掩码（one-hot over 基底展平+警戒区）：改写掩码与之对拍 = 越权写证据
    def inview_onehot(form):
        n0, n1 = BASE_SHAPE[form]
        total = n0 * n1 + GUARD
        if form == "transpose":
            s0, s1, off = 1, V0, 0
        elif form == "slice":
            s0, s1, off = 2 * V1, 2, 0
        elif form == "offset":
            s0, s1, off = V1, 1, V1
        else:
            s0, s1, off = V1, 1, 0
        idx = sorted({off + i * s0 + j * s1 for i in range(V0) for j in range(V1)})
        assert len(idx) == V0 * V1, (form, idx)
        oh = torch.zeros(total, dtype=torch.int64)
        oh[torch.tensor(idx, dtype=torch.int64)] = 1
        return oh

    INVIEW = {f: inview_onehot(f) for f in FORMS}

    def meta_of(t, total):
        # 元数据哨兵：期望形状/步长/偏移/连续性（每案标注期望形状的铁律落点之一）
        return torch.tensor([t.shape[0], t.shape[1], total,
                             t.stride(0), t.stride(1), t.storage_offset(),
                             int(t.is_contiguous()), t.numel()], dtype=torch.int64)

    def reg(name, build, kind, mode, rtol, atol, tag, form, mid, mask_idx, inview_idx):
        cases.append(dict(name=name, build=build, kind=kind, mode=mode, rtol=rtol,
                          atol=atol, tag=tag, form=form, mid=mid,
                          mask_idx=mask_idx, inview_idx=inview_idx,
                          expect=VIEW_EXPECT.get(form, {"view": [4, 3], "note": "#640原貌,无警戒区"})))

    # ---- c000 锚点案：#640 立案原貌逐位复刻（(3,4) 精确存储无警戒区，双列表转置视图就地加） ----
    B0 = torch.randn(3, 4)
    O0 = torch.randn(3, 4)

    def b000():
        inv = torch.ones(12, dtype=torch.int64)

        def run(dev):
            base = B0.to(dev).clone()
            oth = O0.to(dev).clone()
            b0, o0 = base.clone(), oth.clone()          # 纯净副本：期望值/改写掩码在 op 后用它们算
            t, o = base.t(), oth.t()
            torch._foreach_add_([t], [o])
            expected = b0.t() + o0.t()                  # 设备端手工参照（add 主路径三芯已证健康）
            flat_after = base.detach().clone().reshape(-1)
            mask = (flat_after != b0.reshape(-1)).to(torch.int64)
            meta = meta_of(t, 12)
            return [t.detach().clone(), expected, flat_after, mask, inv.to(dev), meta]
        return lambda: run   # build 契约：build() 零参返回 run，run_case 再以 dev 调 run
    reg("c000_repro640_add_transpose_noguard", b000(), "inplace", "exact", 0, 0,
        "ANCHOR-#640原貌(无警戒区)", "transpose", "repro640", 3, 4)

    # ---- 19 成员 × 5 形态矩阵 ----
    def b_cell(form, mid):
        """构造 (member, form) 格的 run 闭包。输入全部来自工厂作用域；run 内零构造零 RNG。"""
        n0, n1 = BASE_SHAPE[form]
        bn = n0 * n1
        total = bn + GUARD
        B = torch.randn(total)                       # 基底 + 尾部警戒区
        O = torch.randn(total)                       # other 列表源（同基底形状→同视图形状）
        T1 = torch.randn(total)                      # addcmul/addcdiv 辅助列表
        T2 = torch.rand(total) * 0.5 + 0.5           # addcdiv 分母避零、量级受控
        inv = INVIEW[form]
        bsh = (n0, n1)

        def run(dev):
            flat = B.to(dev).clone()                 # clone：CPU 侧 .to 返本体，防改工厂张量
            flat_before = flat.detach().clone()
            oflat = O.to(dev).clone()
            base = flat[:bn].view(bsh)
            view = apply_form(base, form)
            if mid == "copy":                        # copy_ 特例：源恒连续（写路径隔离设计）
                oview = oflat[:V0 * V1].view(V0, V1)
            else:
                oview = apply_form(oflat[:bn].view(bsh), form)
            t1 = apply_form(T1.to(dev).clone()[:bn].view(bsh), form)
            t2 = apply_form(T2.to(dev).clone()[:bn].view(bsh), form)
            vs = [view]
            if mid == "zero":
                torch._foreach_zero_(vs)
            elif mid == "copy":
                torch._foreach_copy_(vs, [oview])
            elif mid == "add_list":
                torch._foreach_add_(vs, [oview])
            elif mid == "add_scalar":
                torch._foreach_add_(vs, 1.0)
            elif mid == "mul_scalar":
                torch._foreach_mul_(vs, 1.7)
            elif mid == "sub_scalar":
                torch._foreach_sub_(vs, 0.5)
            elif mid == "div_scalar":
                torch._foreach_div_(vs, 1.5)
            elif mid == "clamp_max":
                torch._foreach_clamp_max_(vs, 0.8)
            elif mid == "clamp_min":
                torch._foreach_clamp_min_(vs, -0.6)
            elif mid == "maximum":
                torch._foreach_maximum_(vs, 0.9)
            elif mid == "exp_":
                torch._foreach_exp_(vs)
            elif mid == "neg_":
                torch._foreach_neg_(vs)
            elif mid == "reciprocal_":
                torch._foreach_reciprocal_(vs)
            elif mid == "lerp_":
                torch._foreach_lerp_(vs, [oview], 0.3)
            elif mid == "addcmul_":
                torch._foreach_addcmul_(vs, [t1], [t2])
            elif mid == "addcdiv_":
                torch._foreach_addcdiv_(vs, [t1], [t2])
            elif mid == "add_oop":
                out = torch._foreach_add(vs, [oview])[0]
                return [out, meta_of(view, total), meta_of(out, total)]
            elif mid == "exp_oop":
                out = torch._foreach_exp(vs)[0]
                return [out, meta_of(view, total), meta_of(out, total)]
            elif mid == "norm":
                try:
                    n = torch._foreach_norm(vs, 2)
                except TypeError:
                    n = torch._foreach_norm(vs)
                return [n[0], meta_of(view, total)]
            else:
                raise ValueError(mid)
            view_after = view.detach().clone()
            flat_after = flat.detach().clone().reshape(-1)
            mask = (flat_after != flat_before).to(torch.int64)
            return [view_after, flat_after, mask, inv.to(dev), meta_of(view, total)]
        return lambda: run   # build 契约：build() 零参返回 run（v2 同款；漏掉此层=全部案"假跑"空转）

    for mid, tag, kind, mode, rtol, atol, note in MEMBERS:
        for form in FORMS:
            name = "m_%s__%s" % (mid, form)
            mi, ii = (2, 3) if kind == "inplace" else (-1, -1)
            reg(name, b_cell(form, mid), kind, mode, rtol, atol,
                "%s%s" % (tag, ("；" + note) if note else ""), form, mid, mi, ii)

    return cases


# ---------------- 序列化 / 比对（对齐 probe_complex_sem.py v2 引擎） ----------------
def ser(t):
    import torch
    t = t.detach().cpu()
    dt = str(t.dtype)   # 先记原始 dtype 再升精度：防"NPU 静默提升 fp32"漏检
    if t.dtype == torch.float16 or t.dtype == torch.bfloat16:
        t = t.float()
    return {"shape": list(t.shape), "dtype": dt, "data": t.tolist()}


def flatten(x):
    if isinstance(x, list):
        for u in x:
            yield from flatten(u)
    else:
        yield x


def run_case(build, dev):
    try:
        outs = build()(dev)
        return {"ok": True, "outs": [ser(t) for t in outs]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": "%s: %s" % (type(e).__name__, e)}


def flat_vals(o):
    return list(flatten(o["data"]))


def compare(case, pcpu, pnpu):
    if not pcpu["ok"] and not pnpu["ok"]:
        return "ERR-SYMM", "双侧均报错(cpu=%s | npu=%s)" % (
            pcpu.get("err", "")[:70], pnpu.get("err", "")[:70])
    if pcpu["ok"] and not pnpu["ok"]:
        return "ERR-ASYMM", "cpu ok, npu err: " + pnpu.get("err", "")[:160]
    if not pcpu["ok"] and pnpu["ok"]:
        return "FAIL-SILENT?", "cpu 报错, npu 静默返回 " + json.dumps(pnpu["outs"])[:160]
    import torch
    ca, na = pcpu["outs"], pnpu["outs"]
    if len(ca) != len(na):
        return "DIVERGE", "输出条数 %d vs %d" % (len(ca), len(na))
    msgs = []
    for i, (x, y) in enumerate(zip(ca, na)):
        if x["dtype"] != y["dtype"]:
            msgs.append("[%d] dtype %s→%s" % (i, x["dtype"], y["dtype"]))
            continue
        if x["shape"] != y["shape"]:
            msgs.append("[%d] shape %s→%s" % (i, x["shape"], y["shape"]))
            continue
        va, vb = flat_vals(x), flat_vals(y)
        a = torch.tensor([float(v) for v in va])
        b = torch.tensor([float(v) for v in vb])
        if case["mode"] == "close" and (case["rtol"] or case["atol"]):
            if not torch.allclose(a, b, rtol=case["rtol"], atol=case["atol"], equal_nan=True):
                d = (a - b).abs().max().item()
                msgs.append("[%d] maxabs=%.3e（rtol=%g atol=%g）" % (i, d, case["rtol"], case["atol"]))
            continue
        eq = (a == b)
        if not bool(eq.all()):
            bad = int((~eq).sum())
            j = int((~eq).nonzero()[0])
            msgs.append("[%d] 位不等 %d 处 e.g. idx%d cpu=%r npu=%r" % (i, bad, j, va[j], vb[j]))
    return ("PASS" if not msgs else "DIVERGE"), "; ".join(msgs)


def refine(case, pcpu, pnpu, verdict):
    """失败形态分类器（仅就地格）：错位写/越权写/静默不生效/纯数值差的显式标注。"""
    if case["kind"] != "inplace" or not (pcpu["ok"] and pnpu["ok"]):
        return verdict, ""
    try:
        cm = flat_vals(pcpu["outs"][case["mask_idx"]])
        ci = flat_vals(pcpu["outs"][case["inview_idx"]])
        nm = flat_vals(pnpu["outs"][case["mask_idx"]])
        ni = flat_vals(pnpu["outs"][case["inview_idx"]])
    except Exception:  # noqa: BLE001
        return verdict, ""
    cpu_oow = sum(1 for m, i in zip(cm, ci) if m > 0 and i < 1)   # CPU 基准自检：应恒 0
    npu_oow = sum(1 for m, i in zip(nm, ni) if m > 0 and i < 1)
    npu_w, cpu_w = sum(1 for m in nm if m > 0), sum(1 for m in cm if m > 0)
    if cpu_oow:
        return verdict, " [CPU基准异常: 视图外改写%d处]" % cpu_oow
    if npu_oow > 0:
        return verdict, " [FORM: GUARD/OOW-WRITE n=%d 越权写视图外/警戒区]" % npu_oow
    if npu_w == 0 and cpu_w > 0:
        return verdict, " [FORM: NOOP-SILENT 静默不生效]"
    if verdict == "DIVERGE":
        if nm == cm:
            # 全覆盖视图上 #640 错序写的掩码与 CPU 相同（写的都是视图内 flat，只是值/序错）——
            # 掩码不可分，用幅值分：ULP 噪声（观测 ~1e-7~1e-5）远小于错位写（O(0.1~10)）
            try:
                va = [float(u) for u in flat_vals(pcpu["outs"][0])]
                vb = [float(u) for u in flat_vals(pnpu["outs"][0])]
                maxabs = max(abs(a - b) for a, b in zip(va, vb)) if va else 0.0
            except Exception:  # noqa: BLE001
                maxabs = float("inf")
            if maxabs > 1e-3:
                return verdict, " [FORM: INVIEW-MISWRITE 区内错值/错序=#640同形 maxabs=%.3g]" % maxabs
            return verdict, " [FORM: VAL-DIFF 掩码同+幅值ULP级 maxabs=%.3g]" % maxabs
        return verdict, " [FORM: INVIEW-MISWRITE 区内错值/错序=#640同形]"
    return verdict, ""


def cellcode(r):
    v = r["verdict"]
    if v == "PASS":
        return "."
    if v == "CHILD-CRASH":
        return "C"
    if v == "TIMEOUT":
        return "T"
    if v == "ERR-ASYMM":
        return "A"
    if v == "ERR-SYMM":
        return "s"
    if v == "FAIL-SILENT?":
        return "F"
    if v == "HARNESS-ERR":
        return "H"
    d = r.get("detail", "")
    for k, letter in (("GUARD/OOW-WRITE", "G"), ("NOOP-SILENT", "N"),
                      ("INVIEW-MISWRITE", "M"), ("VAL-DIFF", "V")):
        if k in d:
            return letter
    return "D"


# ---------------- 子进程 ----------------
def child(idx):
    import torch
    try:
        import torch_npu  # noqa: F401  (真机上此 import 注册 torch.npu；本地冒烟无此模块走 ERR 路径)
    except ImportError:
        pass
    try:
        torch.npu.set_device(0)
    except Exception:
        pass
    case = make_cases()[idx]   # 工厂内构造输入（seed 固定），build/run 零 RNG
    payload = {"cpu": run_case(case["build"], "cpu"),
               DEVICE: run_case(case["build"], DEVICE)}
    print("@@JSON@@" + json.dumps(payload, ensure_ascii=False))


# ---------------- 哨兵 ----------------
def selfcheck():
    """v1 缺陷类回归哨兵（对齐 probe_complex_sem.py v2）：同一子进程连续两次 run_case 且
    不重置 seed——build/run 若重掷 RNG、或原位改写工厂张量/共享状态，两次载荷必不相同 → FAIL。
    就地案的输入 clone 已内置（CPU 侧 .to('cpu') 返本体陷阱），本哨兵作二次把关。"""
    cases = make_cases()
    n_bad = 0
    for c in cases:
        p1 = run_case(c["build"], "cpu")
        p2 = run_case(c["build"], "cpu")   # 刻意不重置 seed
        same = json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)
        if not same:
            n_bad += 1
        print("[%s] rng-invariance %s" % ("PASS" if same else "FAIL", c["name"]))
    print("=" * 72)
    print("selfcheck: %d cases, %d fail（build/run 必须零 RNG、零工厂态污染）" % (len(cases), n_bad))
    sys.exit(1 if n_bad else 0)


# ---------------- 父进程 ----------------
def parent():
    cases = make_cases()
    results = []
    for i, c in enumerate(cases):
        try:
            p = subprocess.run([sys.executable, __file__, "__case__", str(i)],
                               capture_output=True, text=True, timeout=180)
            line = [l for l in p.stdout.splitlines() if l.startswith("@@JSON@@")]
            if not line:
                results.append(dict(name=c["name"], verdict="CHILD-CRASH",
                                    detail=(p.stderr or p.stdout)[-200:], tag=c["tag"],
                                    form=c["form"], mid=c["mid"], expect=c["expect"]))
                continue
            payload = json.loads(line[-1][len("@@JSON@@"):])
            verdict, detail = compare(c, payload["cpu"], payload[DEVICE])
            verdict, form_tag = refine(c, payload["cpu"], payload[DEVICE], verdict)
            results.append(dict(name=c["name"], verdict=verdict,
                                detail=(detail + form_tag).strip(), tag=c["tag"],
                                form=c["form"], mid=c["mid"], expect=c["expect"]))
        except subprocess.TimeoutExpired:
            results.append(dict(name=c["name"], verdict="TIMEOUT", detail="",
                                tag=c["tag"], form=c["form"], mid=c["mid"], expect=c["expect"]))
        except Exception as e:  # noqa: BLE001
            results.append(dict(name=c["name"], verdict="HARNESS-ERR", detail=str(e)[:160],
                                tag=c["tag"], form=c["form"], mid=c["mid"], expect=c["expect"]))
    n_bad = sum(1 for r in results if r["verdict"] in ("DIVERGE", "FAIL-SILENT?", "ERR-ASYMM"))
    n_crash = sum(1 for r in results if r["verdict"] in ("CHILD-CRASH", "TIMEOUT"))
    n_form = sum(1 for r in results if "FORM:" in r.get("detail", ""))
    print("=" * 72)
    for r in results:
        print("[%s] %s | %s" % (r["verdict"], r["name"], r.get("detail", "")))
    print("=" * 72)
    # 矩阵透视表（锚点案单列）
    print("透视表（行=成员/阶梯, 列=形态; . PASS  G 越权写  N 静默不生效  M 区内错位  V 纯值差  "
          "C 崩溃  T 超时  A ERR-ASYMM  s ERR-SYMM  F FAIL-SILENT  D 其它分歧）")
    hdr = "member".ljust(12) + "".join(f.ljust(11) for f in FORMS) + " 阶梯"
    print(hdr)
    by = {(r["mid"], r["form"]): r for r in results}
    for mid, tag, kind, mode, rtol, atol, note in MEMBERS:
        row = mid.ljust(12)
        for f in FORMS:
            r = by.get((mid, f))
            row += (cellcode(r) if r else "?").ljust(11)
        print(row + tag)
    anchor = [r for r in results if r["mid"] == "repro640"]
    for r in anchor:
        print("锚点案: [%s] %s | %s" % (r["verdict"], r["name"], r.get("detail", "")))
    print("=" * 72)
    print("total=%d  findings=%d（DIVERGE/ASYMM/FAIL-SILENT）  crash/timeout=%d  带形态标注=%d"
          % (len(results), n_bad, n_crash, n_form))
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("results ->", RESULT_PATH)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "__case__":
        child(int(sys.argv[2]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "__selfcheck__":
        selfcheck()
    else:
        parent()
