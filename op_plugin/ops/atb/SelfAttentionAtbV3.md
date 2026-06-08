# SelfAttentionAtbV3 算子说明

## 概述

`_npu_flash_attention_v3` 是基于 ATB（Ascend Transformer Boost）的 Flash Attention 算子，使用 `MASK_TYPE_NORM_COMPRESS` 压缩 mask 模式执行自注意力计算。面向 Ascend 310P 场景。

## 与 V1 的关系

V3 基于 V1 (`SelfAttentionAtb.cpp`) 改动，仅新增 `mask_type` 可选参数：

| 维度 | V1 | V3 |
|------|----|----|
| 函数名 | `_npu_flash_attention` | `_npu_flash_attention_v3` |
| maskType | 硬编码 `MASK_TYPE_NORM` | 参数传入，仅支持 `MASK_TYPE_NORM_COMPRESS` |
| 其余配置 | 完全一致 | 完全一致 |
| 输入绑定 | 完全一致 | 完全一致 |
| 底层 ATB 算子 | `SelfAttentionOperation` | `SelfAttentionOperation`（同一个） |

## 与 V2 的对比

V3 与 V2 (`SelfAttentionAtbV2.cpp`) 的差异较大，V3 更接近 V1 的简化版本，而非基于 V2 扩展：

### 总览对比

| 维度 | V2 | V3 |
|------|----|----|
| 函数名 | `_npu_flash_attention_v2` | `_npu_flash_attention_v3` |
| 函数返回值 | `at::Tensor&`（返回 out 引用） | `void`（无返回值） |
| seq_len 类型 | `c10::SymIntArrayRef`（符号整数数组） | `const at::Tensor&`（张量） |
| mask 类型 | `c10::optional<at::Tensor>`（可选） | `const at::Tensor&`（必需） |
| 新增参数 | `kernel_type`, `mask_type`, `slopes` | `mask_type`（可选） |
| mask_type 传入方式 | 必传参数 | 可选参数，默认 NORM_COMPRESS(3) |
| mask_type 支持范围 | NORM, ALIBI 及所有 ALIBI 压缩变体 | 仅 NORM_COMPRESS |
| 输入数量 | 5~6 个（视 mask_type 而定） | 固定 5 个 |
| 目标设备 | 通用 | 面向 Ascend 310P |

### 参数签名对比

```cpp
// V2：参数更多，支持多种 mask 模式
at::Tensor& _npu_flash_attention_v2(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    c10::SymIntArrayRef seq_len,            // 符号整数数组
    const c10::optional<at::Tensor> &mask,  // 可选 mask
    const c10::optional<at::Tensor> &slopes,// ALIBI slopes
    int64_t kernel_type,                    // kernel 类型（必传）
    int64_t mask_type,                      // mask 类型（必传）
    double scale_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    at::Tensor &out
)

// V3：参数更少，仅支持压缩 mask
void _npu_flash_attention_v3(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &mask,                 // 必需 mask
    const at::Tensor &seq_len,              // 张量
    const double scale_value,
    const int64_t num_heads,
    const int64_t num_kv_heads,
    const c10::optional<int64_t> mask_type, // 可选，默认 3
    at::Tensor &out
)
```

### SelfAttentionParam 配置对比

| 配置项 | V2 | V3 |
|--------|----|----|
| `calcType` | `PA_ENCODER` | 与 V1 一致（未显式设置） |
| `kernelType` | 参数传入，支持多种 kernel | `KERNELTYPE_DEFAULT`（固定） |
| `maskType` | 参数传入，支持 NORM/ALIBI 及压缩变体 | 仅 `MASK_TYPE_NORM_COMPRESS` |
| `isTriuMask` | 根据 maskType 动态判断（ALIBI→1, NORM→0） | 固定 `0` |
| `clampType` | `CLAMP_TYPE_UNDEFINED` | 未设置 |
| `clampMin/clampMax` | `0` | 未设置 |
| `windowSize` | `0` | 未设置 |
| `qScale` | `1` | 未设置 |

### 输入绑定（ParamSetter）对比

V2 根据 maskType 条件绑定不同输入：

```cpp
// V2：ALIBI_COMPRESS 模式 → 6 个输入（含 slopes）
if (ALIBI_COMPRESS || ALIBI_COMPRESS_SQRT || ALIBI_COMPRESS_LEFT_ALIGN) {
    Input(query).Input(key).Input(value).Input(mask).Input(seq_len).Input(slopes).Output(out);
}
// V2：ALIBI / NORM 模式 → 5 个输入（不含 slopes）
else if (ALIBI || NORM) {
    Input(query).Input(key).Input(value).Input(mask).Input(seq_len).Output(out);
}
```

V3 固定 5 个输入：

```cpp
// V3：固定 NORM_COMPRESS 模式 → 5 个输入
Input(query, true).Input(key).Input(value).Input(mask).Input(seq_len, true).Output(out);
```

注意 V3 的 `query` 和 `seq_len` 传入了 `true` 参数（表示需要特定处理），而 V2 没有。

### Torch 注册对比

```cpp
// V2：.out 后缀，返回 Tensor(a!)
m.def("_npu_flash_attention_v2.out(
    Tensor query, Tensor key, Tensor value,
    SymInt[] seq_len, *,
    Tensor? mask=None, Tensor? slopes=None,
    int kernel_type=0, int mask_type=2,
    float scale_value=1, int num_heads=0, int num_kv_heads=0,
    Tensor(a!) out) -> Tensor(a!)");

// V3：无 .out 后缀，无返回值
m.def("_npu_flash_attention_v3(
    Tensor query, Tensor key, Tensor value,
    Tensor mask, Tensor seq_len,
    float scale_value, int num_heads, int num_kv_heads,
    *, int ? mask_type = 3,
    Tensor(a!) out) -> ()");
```

### 设计思路对比

| 维度 | V2 | V3 |
|------|----|----|
| **设计理念** | 通用型，支持多种 mask 和 kernel 组合 | 专用型，仅支持压缩 mask |
| **灵活性** | 高，通过参数组合支持多种场景 | 低，硬编码压缩 mask 场景 |
| **复杂度** | 较高，有条件分支逻辑 | 简单，无分支 |
| **适用场景** | 通用 Ascend 平台 | Ascend 310P 专用 |
| **与 V1 的关系** | 基于 V1 扩展，大幅增加参数 | 基于 V1 微调，仅新增 mask_type |

## 调用方式

```python
torch_npu._npu_flash_attention_v3(
    query, key, value, mask, seq_len,
    scale_value, num_heads, num_kv_heads,
    mask_type=3,   # 可选，默认 MASK_TYPE_NORM_COMPRESS(3)
    out=output
)
```

## 函数签名

```cpp
void _npu_flash_attention_v3(
    const at::Tensor &query,              // 查询张量 [total_tokens, num_heads, head_dim]
    const at::Tensor &key,                // Key 张量 [total_tokens, num_kv_heads, head_dim]
    const at::Tensor &value,              // Value 张量 [total_tokens, num_kv_heads, head_dim]
    const at::Tensor &mask,               // 注意力掩码
    const at::Tensor &seq_len,            // 每个序列的长度
    const double scale_value,             // softmax 缩放因子 (1/sqrt(d))
    const int64_t num_heads,              // Query 头数
    const int64_t num_kv_heads,           // KV 头数 (GQA 时 < num_heads)
    const c10::optional<int64_t> mask_type, // mask 类型，仅支持 NORM_COMPRESS(3)
    at::Tensor &out                       // 输出张量（原地写入）
)
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | Tensor | 查询张量，BSND 布局 |
| `key` | Tensor | Key 张量，BSND 布局 |
| `value` | Tensor | Value 张量，BSND 布局 |
| `mask` | Tensor | 注意力掩码 |
| `seq_len` | Tensor | 各序列长度 |
| `scale_value` | double | softmax 缩放因子，通常 `1/sqrt(head_dim)` |
| `num_heads` | int64_t | Query 头数 |
| `num_kv_heads` | int64_t | KV 头数，GQA 时小于 num_heads |
| `mask_type` | optional\<int64_t\> | mask 类型，默认值 3 (`NORM_COMPRESS`)，仅支持该值 |
| `out` | Tensor& | 输出，原地写入 |

## SelfAttentionParam 配置

| 配置项 | 值 | 含义 |
|--------|-----|------|
| `calcType` | `PA_ENCODER` | Encoder 计算模式 |
| `kernelType` | `KERNELTYPE_DEFAULT` | 默认 kernel |
| `maskType` | `MASK_TYPE_NORM_COMPRESS(3)` | 倒三角压缩 mask |
| `kvcacheCfg` | `K_CACHE_V_CACHE` | KV 均有缓存 |
| `scaleType` | `SCALE_TYPE_TOR` | Torch 风格缩放 |
| `quantType` | `TYPE_QUANT_UNDEFINED` | 不使用量化 |
| `cacheType` | `CACHE_TYPE_NORM` | 普通缓存类型 |
| `isTriuMask` | `0` | 不启用上三角因果 mask |
| `inputLayout` | `TYPE_BSND` | 输入布局 [Batch, Seq, Heads, Dim] |

## mask_type 参数说明

```cpp
int64_t mtype = mask_type.has_value() ? mask_type.value() : SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
TORCH_CHECK(mtype == SelfAttentionParam::MASK_TYPE_NORM_COMPRESS, ...);
selfattentionparam.maskType = static_cast<SelfAttentionParam::MaskType>(mtype);
```

- 可选参数，不传时默认 `MASK_TYPE_NORM_COMPRESS(3)`
- 当前仅支持 `MASK_TYPE_NORM_COMPRESS`，传入其他值会触发 `TORCH_CHECK` 报错
- `static_cast<SelfAttentionParam::MaskType>(mtype)` 将 int64 整数转换为枚举类型

## 执行流程

```
1. 解析 mask_type 参数（默认 NORM_COMPRESS）
2. TORCH_CHECK 校验 mask_type 值
3. 构建 SelfAttentionParam（算子参数配置）
4. ParamSetter 绑定 5 个输入 + 1 个输出
   └── Input: query, key, value, mask, seq_len
   └── Output: out
5. OpParamCache 获取/缓存 ATB Operation（参数相同时复用）
6. RunAtbCmd 提交到 NPU 执行
```

## 底层调用路径

```
_npu_flash_attention_v3()
  → OpParamCache<SelfAttentionParam>::getOperation(param, "SelfAttentionOperation")
  → RunAtbCmd(op, paramsetter, "SelfAttentionOperation")
       │
       ▼
  EXEC_ATB_CMD 宏展开：
  ├─ dlopen("libatb.so") 动态加载 ATB 库
  ├─ dlsym("SelfAttentionOperationGetWorkspaceSize") → 获取 workspace 大小
  ├─ dlsym("SelfAttentionOperation") → 执行计算
  ├─ at::Tensor → aclTensor* 类型转换
  └─ 提交到 NPU stream 异步执行
```

## Torch 注册

```cpp
// 算子签名注册
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_flash_attention_v3(
        Tensor query, Tensor key, Tensor value,
        Tensor mask, Tensor seq_len,
        float scale_value, int num_heads, int num_kv_heads,
        *, int ? mask_type = 3,
        Tensor(a!) out) -> ()");
}

// 实现注册（绑定到 NPU 设备）
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_flash_attention_v3",
           TORCH_FN(atb::_npu_flash_attention_v3));
}
```

- `*` 后的参数为关键字参数
- `int ?` 表示可选 int（对应 `c10::optional<int64_t>`）
- `Tensor(a!)` 表示原地修改
- `-> ()` 无返回值，结果写入 `out`

## Python 层注册

`_atb_ops.py` 中 `API_LIST` 添加了 `_npu_flash_attention_v3`，使算子可通过 `torch_npu._npu_flash_attention_v3()` 调用。

## 测试

测试文件：`test/test_custom_ops/test_atb_flash_attention_v3.py`

- 使用 `@SupportedDevices(['Ascend310P'])` 限制仅在 310P 上运行
- 用 numpy 手写 golden 参考计算（softmax + matmul）
- 对比 NPU 算子输出与 golden 结果的相对误差

```bash
python test/test_custom_ops/test_atb_flash_attention_v3.py
```
