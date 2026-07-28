# MM Reduce Scatter XLSX 用例驱动设计

## 背景

`test/test_custom_ops/test_mm_reduce_scatter_base.py` 当前使用固定的
`x1`、`x2` shape，并对同一组输入重复执行 100 次。目标是让脚本从
XLSX 工作簿读取多条 `aclnnMatmulReduceScatter` 用例，根据每行的
`input_tensor_shape`、`input_tensor_range` 和 `input_tensor_dtype`
生成 `x1`、`x2`、`bias`，再通过 PTA 接口
`torch_npu.npu_mm_reduce_scatter_base` 逐 case 验证。

工作簿会在运行前放入 `szg` 容器，因此脚本只接收容器内 Linux 路径，
不直接访问 Windows 的 `D:\...` 路径。

## 目标

- 使用 `openpyxl` 直接读取 `.xlsx` 文件。
- 只初始化一次 HCCL 进程组，随后在所有 rank 上按相同顺序逐 case 执行。
- 支持运行全部用例或按 case 名过滤。
- 支持可配置 `comm_mode`、rank 数和每个 case 的重复次数。
- 第一次迭代完成精度校验，后续迭代用于稳定性或性能观察。
- 对表格格式、shape、dtype 和矩阵维度给出包含 `case_name` 的明确错误。

## 命令行接口

脚本新增以下参数：

- `--case-file`：必填，容器内 `.xlsx` 路径；也允许通过
  `MM_RS_CASE_FILE` 环境变量提供默认值。
- `--sheet`：可选，工作表名称；未指定时读取 active sheet。
- `--case-filter`：可选，对 `case_name` 做子串匹配。
- `--comm-mode`：默认 `aiv`。
- `--world-size`：默认 `4`。
- `--repeat`：默认 `1`。
- `--profile`：可选，仅在显式启用时采集 profiler 数据。

自定义参数在调用 `run_tests()` 前从 `sys.argv` 中移除，避免和 unittest
自身参数冲突。配置会整理为可序列化的普通 Python 数据结构，再传给
spawn 出来的 rank 进程。

## XLSX 解析

必须存在以下表头：

- `op_name`
- `case_name`
- `input_tensor_shape`
- `input_tensor_range`
- `input_tensor_dtype`

只选择 `op_name == "aclnnMatmulReduceScatter"` 的非空行。
三个输入描述字段使用 `ast.literal_eval` 解析，禁止使用 `eval`。

各列表按以下顺序映射：

1. `x1`
2. `x2`
3. `bias`

`input_tensor_dtype` 有三项时创建 bias；只有两项时按 `bias=None`
处理，即使 shape/range 列仍保留第三项。支持的基础 dtype：

- `fp16` / `float16` -> `torch.float16`
- `bf16` / `bfloat16` -> `torch.bfloat16`

range 必须是两个数 `(low, high)`。浮点输入使用均匀分布生成，随后转换到
目标 dtype。随机种子由 case 序号和 rank 派生，保证运行可复现，同时让
不同 rank 使用不同输入。

## Shape 与转置规则

设 `x1.shape == [M, K]`：

- 若 `x2.shape[0] == K`，按 `[K, N]` 直接使用。
- 若 `x2.shape[1] == K`，按 `[N, K]` 读取并执行
  `x2.transpose(0, 1).contiguous()`。
- 否则该 case 校验失败。

启用 bias 时，bias 必须是一维 `[N]`。`M` 必须能被 `world_size`
整除，否则该 case 在执行前失败。

## 分布式执行

主进程读取并校验工作簿，得到 case spec 列表，然后只 spawn
`world_size` 个子进程。每个子进程：

1. 设置对应 NPU device 并初始化 HCCL。
2. 获取 `hcom_name`。
3. 按统一顺序遍历 case spec。
4. 生成当前 rank 的 `x1`、`x2` 和可选 bias。
5. 调用：

   ```python
   torch_npu.npu_mm_reduce_scatter_base(
       x1,
       x2,
       hcom_name,
       world_size,
       reduce_op="sum",
       bias=bias,
       comm_turn=0,
       comm_mode=comm_mode,
   )
   ```

6. 第一次迭代计算参考结果并校验。
7. case 结束后同步所有 rank，并释放当前 case 张量。
8. 所有 case 完成后销毁进程组。

子进程通过队列把 case 结果和异常摘要返回主进程。主进程检查所有子进程
exit code，避免现有代码注释掉 exitcode 检查后出现“子进程失败但测试通过”。

## 精度参考

每个 rank 先计算本地结果：

```python
local = torch.matmul(x1.float(), x2.float())
if bias is not None:
    local = local + bias.float()
```

然后对 `local` 执行 `dist.all_reduce(..., SUM)`，再按 rank 从 M 维切出
`M / world_size` 行，转换为算子输出 dtype，与融合算子结果比较。

默认沿用当前测试的 `rtol=0.05`、`atol=0.05`。

## Profiling

当前文件中的 profiling 代码存在缩进错误，而且未调用 `prof.step()`。
修改后 profiling 默认关闭；传入 `--profile` 时，仅采集首个选中 case
的第一次算子调用，并正确调用 `prof.step()`。输出目录按 case 名区分，
防止多个 case 覆盖同一份结果。

## 错误处理与输出

- 工作簿不存在、扩展名不是 `.xlsx`、sheet 不存在或表头缺失：测试启动前失败。
- 单元格解析失败：报告 Excel 行号、case 名和列名。
- dtype、range 或 shape 非法：报告具体 case 和输入名。
- 某个 rank 执行失败：返回主进程并令测试失败。
- 最终打印通过、失败、跳过数量和失败 case 摘要。

## 测试策略

先为纯 Python 辅助逻辑编写失败测试，再实现：

- XLSX 表头及行解析。
- 两项 dtype 对应无 bias。
- 三项 dtype 对应有 bias。
- `[K,N]` 与 `[N,K]` 两种 x2 布局。
- 非法 dtype、range、shape 和过滤后无 case。
- CLI 参数与 unittest 参数分离。

随后执行：

- `py_compile` 语法检查。
- 不访问 NPU 的解析单元测试。
- 使用一个小型 XLSX case 做 4 卡 NPU 冒烟测试。
- 选取有 bias、无 bias、x2 转置和 bf16/fp16 代表 case 做验证。

## 非目标

- 不读取旧版 `.xls`。
- 不在脚本中访问 Windows 本地路径或自动上传工作簿。
- 不处理量化 `int8` case；现有 quant 测试入口保持独立。
- 不修改 PTA 算子实现或通信库。
