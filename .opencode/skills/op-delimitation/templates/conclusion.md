# Step 5 结论报告模板

输出文件：`{opName}_pta_atk/分析结论.md`（命名规则与 [report SKILL.md](../report/op-delimitation-report/SKILL.md#产出位置硬约束) 对齐）

必须包含以下 8 项：

```markdown
## 问题：[opName] [问题类型]

### 结论
[结论类型：算子问题 | 代码 bug（已修复） | 代码 bug（待修复） | 用例问题 | 无法复现 | 环境/配置问题]

### 关键报错信息
[从日志中摘录的原始报错信息，保留完整堆栈或错误码]

### op-atk 4 步执行记录
① **auto_adapt.py 输出**：opName = `<opName>`；接口原型 = `[签名 + 参数列表]`（来自 `_op_plugin_docs.py`）；@register api_type = `<api_type>`（= atk.yaml name）
② **atk case 生成**：`{opName}_pta_atk/result/<atk>.json` 路径（3 件套模式：跳过——用户提供 case.json，记录 case.json 路径）
③ **executor.py npu 分支改写摘要**：在 @register 类加 `elif self.device == "npu"` 分支；tensor 输入 `deepcopy + .to(self.set_device)`；标量透传；调 `torch_npu.<opName>(...)`
④ **atk task 命令 + tee 路径 + xlsx 路径**：
   - 命令：`atk task -c <case json 按模式：A=result/<atk>.json；B=传入的 case.json> -p executor.py -n node.yaml [-e 50]`
   - tee 日志：`运行日志/pta_atk_<opName>_<ts>.log`
   - xlsx：`atk_output/all_<api_type>_<ts>/report/*.xlsx`（**CaseSet = api_type，不是 opName**——按 api_type 拼路径）

### 定位过程
- UT 验证结果：[max diff 数值、确定性结果]
- aclnn 算子调用列表：[去重后的算子]
- Dump 分析结果：[首个发散算子 + 差异值]（如有）
- 历史参考：[类似案例编号]（如有）

### 根因分析
[代码行号或算子名称 + 判定依据]

### 设备环境信息
（见 [device-env-snapshot](device-env-snapshot.md) 块；CANN/torch/torch_npu/驱动/NPU + `op_plugin_repo_root`/`op_plugin_git_commit`）

### 修复方案（如有 Step 4）
[代码 diff 或修复描述，引用 step4_fix.md]
```
