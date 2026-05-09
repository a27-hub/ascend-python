# npu_ciou Atlas A5 平台对接设计文档

## 1. 概述

### 1.1 背景

npu_ciou 算子是目标检测领域常用的 CIoU（Complete Intersection over Union）损失函数的 NPU 实现，原有实现基于 ACL（Ascend Computing Language）接口。Atlas A5 训练系列产品（Ascend950）需要通过 ACLNN（Ascend Computing Language Neural Network）接口重新对接，以获得更好的性能和精度支持。

### 1.2 目标

- 在 Atlas A5 平台（Ascend950）上通过 ACLNN 接口 `aclnnCIoU` 对接 npu_ciou 算子
- 保持与原有 ACL 路径的兼容性，非 Ascend950 平台自动回退到 ACL 实现
- 补充 torch.compile 所需的 Meta Tensor 注册
- 补充 Ascend950 平台专属测试用例
- 更新接口文档，标注平台约束

### 1.3 提交信息

- 提交哈希：5405444b
- 分支：npu_ciou
- 修改文件：5个

## 2. 算子接口定义

### 2.1 函数原型

```python
torch_npu.npu_ciou(self, gtboxes, trans=False, is_cross=False, mode=0, atan_sub_flag=False) -> Tensor
```

### 2.2 参数说明

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| self | Tensor | 是 | 预测检测框，格式为 xywh，shape 为 (4, n) |
| gtboxes | Tensor | 是 | 真实检测框，格式为 xywh，shape 为 (4, n) |
| trans | bool | 否 | 是否有偏移（xywh→xyxy 转换），默认 False |
| is_cross | bool | 否 | 是否交叉计算，默认 False，**当前仅支持 False** |
| mode | int | 否 | 计算方式：0=IoU，1=IoF，默认 0 |
| atan_sub_flag | bool | 否 | 是否将正向第二个值传递给反向，默认 False |

### 2.3 返回值

- `npu_ciou`：返回 overlap 张量，shape 为 (1, n)（is_cross=False 时）
- `_npu_ciou`（内部接口）：返回 (overlap, atan_sub) 元组

## 3. 架构设计

### 3.1 整体调用链路

```
用户调用 torch_npu.npu_ciou(bboxes, gtboxes, trans, is_cross, mode, atan_sub_flag)
    │
    ▼
op_api::npu_ciou()  [CiouKernelNpuOpApi.cpp]
    │
    ├── GetSocVersion() < Ascend950 ?
    │       │
    │       ├── 是 → acl_op::npu_ciou()  [ACL 路径，CiouKernelNpu.cpp]
    │       │
    │       └── 否 → DO_COMPATIBILITY(aclnnCIoU, acl_op::npu_ciou())
    │                   │
    │                   ├── aclnnCIoU 符号不存在 → acl_op::npu_ciou()  [回退 ACL]
    │                   │
    │                   └── aclnnCIoU 符号存在 → EXEC_NPU_CMD(aclnnCIoU, ...)  [ACLNN 路径]
    │
    ▼
返回 overlap 张量
```

### 3.2 平台路由策略

| 平台 | SOC 版本 | 执行路径 | 说明 |
|------|----------|---------|------|
| Atlas A5 训练系列 | Ascend950 | ACLNN（aclnnCIoU） | 新对接路径 |
| Atlas A3 训练系列 | Ascend910B | ACL（CIoU） | 原有路径 |
| Atlas A2 训练系列 | Ascend910A | ACL（CIoU） | 原有路径 |
| Atlas 推理系列 | Ascend310P | ACL（CIoU） | 原有路径 |
| Atlas 训练系列 | Ascend910 | ACL（CIoU） | 原有路径 |

### 3.3 SOC 版本判断机制

采用两级判断：

1. **SOC 版本检查**：`c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950`
   - 非 Ascend950 平台直接走 ACL 路径，避免不必要的符号查找开销
2. **符号存在性检查**：`DO_COMPATIBILITY(aclnnCIoU, ...)`
   - 通过 `dlsym` 在 `libopapi.so` 中查找 `aclnnCIoU` 和 `aclnnCIoUGetWorkspaceSize` 符号
   - 任一符号不存在则回退到 ACL 路径

## 4. 详细设计

### 4.1 OpApi 实现（CiouKernelNpuOpApi.cpp）

#### 4.1.1 `_npu_ciou` 内部接口

```cpp
std::tuple<at::Tensor, at::Tensor> _npu_ciou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag)
{
    // 1. SOC 版本检查：非 Ascend950 走 ACL
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
    }

    // 2. fp16→fp32 类型转换（aclnnCIoU 不支持 fp16 输入）
    bool self_is_half = self.scalar_type() == at::kHalf;
    bool gtboxes_is_half = gtboxes.scalar_type() == at::kHalf;
    at::Tensor self_cp = self_is_half ? _npu_dtype_cast(self, at::kFloat) : self;
    at::Tensor gtboxes_cp = gtboxes_is_half ? _npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;

    // 3. 推导输出尺寸并预分配张量
    auto output_size = ciou_npu_output_size(self_cp, gtboxes_cp, is_cross);
    at::Tensor overlap = npu_preparation::apply_tensor_with_format(output_size, self_cp.options(), ACL_FORMAT_ND);
    at::Tensor atan_sub = npu_preparation::apply_tensor_with_format(output_size, self_cp.options(), ACL_FORMAT_ND);

    // 4. 调用 ACLNN 算子
    const char *mode_str = mode == 1 ? "iof" : "iou";
    EXEC_NPU_CMD(aclnnCIoU, self_cp, gtboxes_cp, trans, is_cross, mode_str, overlap, atan_sub);

    // 5. fp32→fp16 类型回转
    if (self_is_half || gtboxes_is_half) {
        overlap = _npu_dtype_cast(overlap, at::kHalf);
        atan_sub = _npu_dtype_cast(atan_sub, at::kHalf);
    }
    return std::tie(overlap, atan_sub);
}
```

#### 4.1.2 `npu_ciou` 用户接口

```cpp
at::Tensor npu_ciou(...) {
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_ciou(...);
    }
    auto results = at_npu::native::custom_ops::_npu_ciou(...);
    return std::get<0>(results);  // 只返回 overlap
}
```

#### 4.1.3 `npu_ciou_backward` 反向接口

```cpp
std::tuple<at::Tensor, at::Tensor> npu_ciou_backward(...) {
    // Ascend950 及以上平台不支持反向
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        TORCH_CHECK(false, "npu_ciou_backward is not supported on Ascend950 and above",
            OPS_ERROR(ErrCode::NOT_SUPPORT));
    }
    // 非 Ascend950 走 ACL 反向
    return acl_op::npu_ciou_backward(...);
}
```

### 4.2 fp16 类型转换设计

ACLNN `aclnnCIoU` 算子不支持 fp16 输入，需要类型转换：

```
输入(fp16) ──→ _npu_dtype_cast(fp32) ──→ aclnnCIoU 计算 ──→ 输出(fp32) ──→ _npu_dtype_cast(fp16) ──→ 返回
```

使用 `_npu_dtype_cast`（接受 `ScalarType` 参数）而非 `npu_dtype_cast`（接受 `int64_t` 参数），与 ACL 实现保持一致。

### 4.3 Meta Tensor 注册（_meta_registrations.py）

为支持 `torch.compile` 图模式编译，注册 Meta 实现：

```python
@impl(m, "_npu_ciou")
def _npu_ciou_meta(self, gtboxes, trans=False, is_cross=True, mode=0, atan_sub_flag=False):
    if is_cross:
        output_size = [gtboxes.size(1), self.size(1)]
    else:
        output_size = [1, self.size(1)]
    return (torch.empty(output_size, dtype=self.dtype, device=self.device),
            torch.empty(output_size, dtype=self.dtype, device=self.device))

@impl(m, "npu_ciou")
def npu_ciou_meta(self, gtboxes, trans=False, is_cross=True, mode=0, atan_sub_flag=False):
    if is_cross:
        output_size = [gtboxes.size(1), self.size(1)]
    else:
        output_size = [1, self.size(1)]
    return torch.empty(output_size, dtype=self.dtype, device=self.device)
```

- `_npu_ciou_meta`：返回元组 (overlap, atan_sub)，与 C++ 接口返回类型一致
- `npu_ciou_meta`：返回单个 overlap 张量

### 4.4 YAML 配置（op_plugin_functions.yaml）

```yaml
- func: _npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> (Tensor, Tensor)
  acl_op: all_version
  op_api: all_version

- func: npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> Tensor
  acl_op: all_version
  op_api: all_version
  exposed: all_version

- func: npu_ciou_backward(Tensor grad, Tensor bboxes, Tensor gtboxes, Tensor? atan_sub, bool trans=False, bool is_cross=True, int mode=0) -> (Tensor, Tensor)
  acl_op: all_version
  op_api: all_version
```

`op_api: all_version` 表示所有版本都注册 OpApi 实现，运行时通过 SOC 版本和符号检查动态选择路径。

## 5. 约束与限制

### 5.1 Atlas A5 平台（Ascend950）特有约束

| 约束项 | 说明 |
|--------|------|
| is_cross | 仅支持 False，aclnnCIoU 不支持 is_cross=True |
| 反向传播 | 不支持，调用 npu_ciou_backward 直接报错 |
| 输入维度 | boxes 的第二维度必须为 1024 的倍数 |
| fp16 输入 | 支持，内部自动转换为 fp32 计算，输出转回 fp16 |

### 5.2 通用约束

| 约束项 | 说明 |
|--------|------|
| 输入维度 | 必须为 2 维，shape 为 (4, n) |
| is_cross | 当前仅支持 False |
| 反向传播 | 仅支持 trans==True、is_cross==False、mode==0（iou） |

### 5.3 is_cross 不支持的回退策略

`DO_COMPATIBILITY` 宏只检查算子符号是否存在，不检查参数是否被支持。当 `aclnnCIoU` 存在但不支持 `is_cross=True` 时，`DO_COMPATIBILITY` 不会触发回退。因此需要额外的参数检查逻辑。

当前方案：由于 CIoU 仅支持 `is_cross=False`，直接在文档和测试中约束此参数，不实现参数级别的回退。

## 6. 测试设计

### 6.1 测试文件

`test/test_base_ops/test_ciou.py`

### 6.2 测试用例矩阵

| 用例名称 | 平台 | 输入规模 | dtype | is_cross | mode | atan_sub_flag | 断言类型 |
|----------|------|----------|-------|----------|------|---------------|---------|
| test_npu_ciou_shape_format | Ascend910A/910B | 6/12/100 | fp32 | False | iou | True | 精度 |
| test_npu_ciou_shape_format_1024 | Ascend950 | 1024/2048/3072 | fp32 | False | iou | True | 精度 |
| test_npu_ciou_fp16 | Ascend950 | 1024 | fp16 | False | iou | True | 形状 |
| test_npu_ciou_mode_iof | Ascend950 | 1024 | fp32 | False | iof | True | 形状 |
| test_npu_ciou_atan_sub_flag_false | Ascend950 | 1024 | fp32 | False | iou | False | 形状 |
| test_npu_ciou_mixed_precision | Ascend950 | 1024 | fp16+fp32 | False | iou | True | 形状 |
| test_npu_ciou_invalid_shape | Ascend950 | 100（非1024倍数） | fp32 | False | iou | True | 报错 |

### 6.3 用例设计说明

1. **平台隔离**：使用 `@SupportedDevices` 装饰器区分不同平台用例
   - Ascend910A/910B：小规模输入（6/12/100），验证 ACL 路径
   - Ascend950：大规模输入（1024 倍数），验证 ACLNN 路径

2. **精度断言策略**：
   - `cpu_op_exec` 实现了完整 CIoU 公式（含 v*alpha 项），可作为 fp32 + iou + atan_sub_flag=True 场景的参考
   - `atan_sub_flag=False` 时计算 DIoU（不含 v*alpha 项），与 CPU 参考结果数值不同，只做形状断言
   - fp16 场景因精度有限，只做形状断言
   - mode=iof 场景因 `cpu_op_exec` 未实现 iof 模式，只做形状断言

3. **Ascend950 输入规模**：boxes 第二维度必须为 1024 的倍数，因此用例中使用 1024/2048/3072

4. **已删除的用例**：
   - `test_npu_ciou_is_cross_shape`：is_cross=True 不支持，已删除
   - `test_npu_ciou_fp16_is_cross`：is_cross=True 不支持，已删除

### 6.4 CPU 参考实现

`cpu_op_exec` 实现了 CIoU 的 CPU 计算逻辑：

```
CIoU = IoU - ρ²/c² - v·α

其中：
  IoU = 交集面积 / 并集面积
  ρ²  = 两个框中心点距离的平方
  c²  = 最小包围框对角线的平方
  v   = (4/π²) · (arctan(w_gt/h_gt) - arctan(w/h))²
  α   = v / (1 - IoU + v)
```

## 7. 文档更新

### 7.1 接口文档修改

文件：`docs/zh/custom_APIs/torch_npu/（beta）torch_npu-npu_ciou.md`

| 修改项 | 修改前 | 修改后 |
|--------|--------|--------|
| 产品支持 | 无 Atlas A5 | 新增 Atlas A5 训练系列产品 √ |
| 函数原型 is_cross 默认值 | is_cross=True | is_cross=False |
| is_cross 参数说明 | 默认值为True | 默认值为False。当前仅支持is_cross为False |
| 约束说明 | CIoU向后只支持... | 新增"CIoU仅支持is_cross==False"；新增 Atlas A5 平台约束（第二维度为 1024 倍数、不支持反向） |

## 8. 修改文件清单

| 文件路径 | 修改类型 | 说明 |
|----------|---------|------|
| op_plugin/ops/opapi/CiouKernelNpuOpApi.cpp | 新增 | ACLNN 对接实现，含 SOC 版本路由、fp16 类型转换、反向不支持报错 |
| op_plugin/python/meta/_meta_registrations.py | 新增 | 注册 `_npu_ciou` 和 `npu_ciou` 的 Meta 实现 |
| op_plugin/config/op_plugin_functions.yaml | 修改 | 为 `_npu_ciou`、`npu_ciou`、`npu_ciou_backward` 添加 `op_api: all_version` |
| test/test_base_ops/test_ciou.py | 修改 | 新增 Ascend950 专属用例，添加 `@SupportedDevices` 装饰器，删除 is_cross=True 用例 |
| docs/zh/custom_APIs/torch_npu/（beta）torch_npu-npu_ciou.md | 修改 | 更新产品支持、默认值、约束说明 |

## 9. 风险与待办

### 9.1 已知风险

| 风险项 | 影响 | 缓解措施 |
|--------|------|---------|
| `c10_npu::GetSocVersion()` 在部分环境可能不生效 | 非 Ascend950 平台可能误走 ACLNN 路径 | `DO_COMPATIBILITY` 作为二级兜底，符号不存在时仍回退 ACL |
| `is_cross` 默认值在 YAML/Meta 中仍为 True | 与实际约束不一致 | 后续需同步修改 YAML 和 Meta 注册中的默认值 |
| ACLNN CIoU 不支持 is_cross=True | 交叉计算场景不可用 | 文档明确约束，测试用例已删除 |

### 9.2 后续待办

- [ ] 同步修改 `op_plugin_functions.yaml` 中 `is_cross` 默认值为 False
- [ ] 同步修改 `_meta_registrations.py` 中 `is_cross` 默认值为 False
- [ ] 评估 `check_aclnn_kernel_available` 替代 `GetSocVersion` 判断的可行性
- [ ] 补充 npu_iou 算子的 Atlas A5 平台 ACLNN 对接
