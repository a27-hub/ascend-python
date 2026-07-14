# torch_npu.npu_fused_infer_attention_score

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- **API功能**：适配增量&全量推理场景的 FlashAttention 算子。当 `query` 矩阵的 S=1，进入 IncreFlashAttention（decode）分支；其余场景进入 PromptFlashAttention（prefill）分支。

- **两种计算场景**：

  | 场景 | 触发条件 | 说明 |
  | ---- | -------- | ---- |
  | PromptFlashAttention（prefill） | Q_S > 1 | `query/key/value` 为整段序列张量。关注吞吐，长序列需关注超时风险与切分策略。 |
  | IncreFlashAttention（decode） | Q_S = 1 | `query` 的 S 维为 1，`key/value` 通常来自 KV Cache。关注单步延迟与 KV Cache 访问效率。 |

- **计算公式**：

  根据传入参数不同，算子自动选择以下公式分支：

  | 公式分支 | 触发条件 |
  | -------- | -------- |
  | [基本公式](#basic_formula) | 不传量化参数、不传 rope、不传伪量化参数 |
  | [位置编码公式](#pse_formula) | 传入 `pse_shift` |
  | [全量化公式](#full_quant_formula) | 传入 `dequant_scale1` + `quant_scale1` + `dequant_scale2` + `quant_scale2`，输入/输出为 int8 |
  | [后量化公式](#post_quant_formula) | 传入 `quant_scale2`，仅输出为 int8 |
  | [伪量化公式](#pseudo_quant_formula) | 传入 `antiquant_scale` / `key_antiquant_scale` 等，key/value 为 int8/int4 |
  | [MLA 公式](#mla_formula) | 传入 `query_rope` + `key_rope` |

  <a id="basic_formula"></a>**基本公式**：

  $$
  attention\_out = softmax \left(scale \cdot (query \cdot key^\top) + atten\_mask \right) \cdot value
  $$

  <a id="pse_formula"></a>**位置编码**：

  $$
  attention\_out = softmax\left( scale \cdot (query \cdot key^\top) + pse\_shift + atten\_mask \right) \cdot value
  $$

  <a id="full_quant_formula"></a>**全量化公式**：

  $$
  query_{fp} = query \cdot dequant\_scale1
  $$

  $$
  key_{fp} = key \cdot dequant\_scale1
  $$

  $$
  value_{fp} = value \cdot dequant\_scale1
  $$

  $$
  score_{int8} = quantize(scale \cdot (query_{fp} \cdot key_{fp}^\top) + mask, \; quant\_scale1)
  $$

  $$
  attention\_out = quantize(softmax(score_{int8}) \cdot value_{fp}, \; quant\_scale2, \; quant\_offset2)
  $$

  <a id="post_quant_formula"></a>**后量化公式**：

  $$
  attention\_out = quantize(softmax(scale \cdot (query \cdot key^\top) + mask) \cdot value, \; quant\_scale2, \; quant\_offset2)
  $$

  <a id="pseudo_quant_formula"></a>**伪量化公式**：

  $$
  key_{fp} = key \cdot antiquant\_scale + antiquant\_offset
  $$

  $$
  value_{fp} = value \cdot antiquant\_scale + antiquant\_offset
  $$

  $$
  attention\_out = softmax(scale \cdot (query \cdot key_{fp}^\top) + mask) \cdot value_{fp}
  $$

  <a id="mla_formula"></a>**MLA 场景**：

  $$
  query_{rot} = RoPE(query, query\_rope)
  $$

  $$
  key_{rot} = RoPE(key, key\_rope)
  $$

  $$
  attention\_out = softmax(scale \cdot (query_{rot} \cdot key_{rot}^\top) + mask) \cdot value
  $$

  其中，`quantize` 为量化函数，`RoPE` 为旋转位置编码函数。

## 参数快速参考

> **维度约定**：B = Batch Size, S = Sequence Length, H = Hidden Size, N = Head Num, D = Head Dim（满足 D = H / N）, T = 所有 Batch 序列长度的累加和。
> Q_S/S1 = query 的 S, KV_S/S2 = key/value 的 S, Q_N = num_query_heads, KV_N = num_key_value_heads。

### 按场景分类的参数速查

| 场景 | 必选参数 | 关键可选参数 | 对应约束章节 |
| ---- | -------- | ------------ | ------------ |
| 基础 Prefill（MHA） | `query, key, value` | `atten_mask, num_heads, scale, input_layout` | [通用基础约束](#base_constraints), [Q_S > 1 约束](#qs_gt_1) |
| 基础 Decode（MHA） | `query, key, value` | `atten_mask, num_heads, scale, input_layout, actual_seq_lengths_kv` | [通用基础约束](#base_constraints), [Q_S = 1 约束](#qs_eq_1) |
| Decode + PageAttention | `query, key, value, block_table, block_size` | `actual_seq_lengths_kv` | [PageAttention 约束](#pa_constraints) |
| GQA | `query, key, value, num_heads, num_key_value_heads` | — | [通用基础约束](#base_constraints) |
| MLA | `query, key, value, query_rope, key_rope` | `num_heads, scale, input_layout` | [MLA 约束](#mla_constraints) |
| Prefix（共享前缀） | `query, key, value, key_shared_prefix, value_shared_prefix` | `actual_shared_prefix_len` | [Prefix 约束](#prefix_constraints) |
| int8 全量化 | `query, key, value, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2` | `quant_offset2` | [int8 量化约束](#int8_constraints) |
| int8 后量化 | `query, key, value, quant_scale2` | `quant_offset2` | [int8 量化约束](#int8_constraints) |
| 伪量化（KV int8→fp16） | `query, key, value, antiquant_scale` | `antiquant_offset, antiquant_mode` | [伪量化约束](#pseudo_quant_constraints) |
| 伪量化分离模式 | `query, key, value, key_antiquant_scale, value_antiquant_scale` | `key_antiquant_mode, value_antiquant_mode` | [Q_S > 1 约束](#qs_gt_1) / [Q_S = 1 约束](#qs_eq_1) |
| 左 Padding | `query, key, value, actual_seq_lengths, query_padding_size` | `kv_padding_size` | [Padding 约束](#padding_constraints) |
| Ring Attention | `query, key, value, softmax_lse_flag=True` | — | [返回值说明](#return_value) |
| TND 布局 | `query, key, value, actual_seq_lengths, actual_seq_lengths_kv, input_layout="TND"` | — | [TND 布局约束](#tnd_constraints) |

### 各轴取值范围速查

| 轴 | 含义 | Q_S > 1 上限 | Q_S = 1 上限 | 对齐要求 |
| -- | ---- | ------------ | ------------ | -------- |
| B | Batch Size | ≤ 65536 | ≤ 65536 | — |
| N (Q_N) | query head 数 | ≤ 256 | ≤ 256 | — |
| N (KV_N) | key/value head 数 | ≤ 256 | ≤ 256 | — |
| D | Head Dim | ≤ 512 | ≤ 512 | fp16/bf16: 16 对齐; int8: 32 对齐; int4: 64 对齐 |
| S (Q_S) | query 序列长度 | ≤ 20971520 (20M) | = 1 | — |
| S (KV_S) | key/value 序列长度 | ≤ 20971520 (20M) | ≤ 262144 | — |

## 函数原型

```python
torch_npu.npu_fused_infer_attention_score(
    query, key, value, *,
    pse_shift=None, atten_mask=None,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    dequant_scale1=None, quant_scale1=None,
    dequant_scale2=None, quant_scale2=None, quant_offset2=None,
    antiquant_scale=None, antiquant_offset=None,
    block_table=None, query_padding_size=None, kv_padding_size=None,
    key_antiquant_scale=None, key_antiquant_offset=None,
    value_antiquant_scale=None, value_antiquant_offset=None,
    key_shared_prefix=None, value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None, key_rope=None, key_rope_antiquant_scale=None,
    num_heads=1, scale=1.0,
    pre_tokens=2147483647, next_tokens=2147483647,
    input_layout="BSH", num_key_value_heads=0,
    sparse_mode=0, inner_precise=0,
    block_size=0, antiquant_mode=0,
    softmax_lse_flag=False,
    key_antiquant_mode=0, value_antiquant_mode=0
) -> (Tensor, Tensor)
```

> **说明**：`*` 之前的 `query, key, value` 为位置参数（必须按顺序传入）；`*` 之后的为关键字参数（键值对赋值，可选，不传使用默认值）。

## 参数说明

### input_layout 枚举说明

`input_layout` 标识输入 `query, key, value` 的数据排布格式。**格式中带下划线时**，下划线左边为输入 layout，右边为输出 layout，算子内部自动 layout 转换。

| input_layout | 输入 shape（query） | 输入 shape（key/value） | 输出 shape | 适用场景 |
| ------------ | ------------------- | ----------------------- | ---------- | -------- |
| `"BSH"` | (B, Q_S, H) | (B, KV_S, H) | (B, Q_S, H) | 通用，H = N × D |
| `"BSND"` | (B, Q_S, Q_N, D) | (B, KV_S, KV_N, D) | (B, Q_S, Q_N, D) | 通用，N/D 分离 |
| `"BNSD"` | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, Q_N, Q_S, D) | 通用，N 轴在前 |
| `"BNSD_BSND"` | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, Q_S, Q_N, D) | 仅 Q_S > 1 |
| `"BSH_NBSD"` | (B, Q_S, H) | (B, KV_S, H) | (B, Q_N, Q_S, D) | layout 转换 |
| `"BSND_NBSD"` | (B, Q_S, Q_N, D) | (B, KV_S, KV_N, D) | (B, Q_N, Q_S, D) | layout 转换 |
| `"BNSD_NBSD"` | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, Q_N, Q_S, D) | 仅 Q_S > 1 且 Q_S ≤ 16 |
| `"TND"` | (T, Q_N, D) | (T, KV_N, D) | (T, Q_N, D) | 变长序列，T = Σseqlen |
| `"TND_NTD"` | (T, Q_N, D) | (T, KV_N, D) | (Q_N, T, D) | 变长序列 + transpose |
| `"NTD_TND"` | (Q_N, T, D) | (KV_N, T, D) | (T, Q_N, D) | 变长序列 + transpose |

### sparse_mode 枚举说明

| sparse_mode | 模式 | 说明 | atten_mask 要求 |
| ----------- | ---- | ---- | --------------- |
| 0 | defaultMask | 默认模式。不传 `atten_mask` 时不做 mask；传入时使用完整 mask 矩阵，`pre_tokens`/`next_tokens` 之间的部分参与计算。Q_S=1 且不带 rope 时此参数无效。 | shape：(Q_S, KV_S) 或 (1, Q_S, KV_S) 等 |
| 1 | allMask | 必须传入完整 `atten_mask` 矩阵 (S1×S2)。忽略 `pre_tokens`/`next_tokens`。 | shape：(Q_S, KV_S) 或 (1, Q_S, KV_S) 等 |
| 2 | leftUpCausal | 左上角 causal mask。需传入优化后的 `atten_mask` 矩阵。 | shape：(2048, 2048) 或 (1, 2048, 2048) 或 (1, 1, 2048, 2048) |
| 3 | rightDownCausal | 右下角 causal mask（以右顶点为划分的下三角）。需传入优化后的 mask。 | shape：(2048, 2048) 或 (1, 2048, 2048) 或 (1, 1, 2048, 2048) |
| 4 | band | band 模式 mask。需传入优化后的 `atten_mask` 矩阵。 | shape：(2048, 2048) 或 (1, 2048, 2048) 或 (1, 1, 2048, 2048) |
| 5 | prefix | **暂不支持** | — |
| 6 | global | **暂不支持** | — |
| 7 | dilated | **暂不支持** | — |
| 8 | block_local | **暂不支持** | — |

> **特殊约束**：Q_S = 1 且不带 rope 输入时，`sparse_mode` 参数无效。

### inner_precise 枚举说明

共 4 种模式，使用两个 bit 位：bit0 = 高精度(0)/高性能(1)，bit1 = 是否行无效修正(0=否, 1=是)。

| inner_precise | 精度模式 | 行无效修正 | 适用场景 |
| ------------- | -------- | ---------- | -------- |
| 0 | 高精度 | 否 | 默认，精度优先 |
| 1 | 高性能 | 否 | 性能优先 |
| 2 | 高精度 | 是 | mask 存在整行全 1 时提升精度 |
| 3 | 高性能 | 是 | mask 存在整行全 1 时提升精度（性能略降） |

> **说明**：bf16 和 int8 不区分高精度/高性能（bit0 无效）。行无效修正对 fp16、bf16、int8 均生效。当 mask 存在整行全 1 时精度可能损失，可尝试配置 2 或 3。Q_S = 1 时仅支持 0 和 1。若算子可判断出存在无效行场景（如 sparse_mode=3 且 Sq > Skv），会自动开启行无效计算。

### key_antiquant_mode / value_antiquant_mode 枚举说明

| 值 | 模式 | 说明 |
| -- | ---- | ---- |
| 0 | perchannel | per-channel 模式（包含 per-tensor）。默认值。 |
| 1 | pertoken | per-token 模式。 |
| 2 | pertensor + perhead | per-tensor 叠加 per-head 模式。 |
| 3 | pertoken + perhead | per-token 叠加 per-head 模式。 |
| 4 | pertoken + page attention | per-token 叠加 page attention 管理 scale/offset。 |
| 5 | pertoken + perhead + page attention | per-token 叠加 per-head 并使用 page attention 管理 scale/offset。 |

> **特殊约束**：

> - Q_S ≥ 2 时仅支持 0、1。
> - Q_S = 1 时支持 0~5。
> - 除 `key_antiquant_mode=0` 且 `value_antiquant_mode=1` 的场景外，`key_antiquant_mode` 与 `value_antiquant_mode` 必须一致。

### 详细参数说明

#### 必选参数（位置参数）

- **query** (`Tensor`)：attention 的 Query 输入。
  - 数据类型：`float16`、`bfloat16`
  - 数据格式：ND
  - 不支持非连续 Tensor。

- **key** (`Tensor`)：attention 的 Key 输入。
  - 数据类型：`float16`、`bfloat16`、`int8`、`int4`（`int32` 容器，即 8 个 int4 拼接为一个 int32）
  - 数据格式：ND
  - 不支持非连续 Tensor。

- **value** (`Tensor`)：attention 的 Value 输入。
  - 数据类型：`float16`、`bfloat16`、`int8`、`int4`（`int32` 容器）
  - 数据格式：ND
  - 不支持非连续 Tensor。

#### 位置编码参数

- **pse_shift** (`Tensor`, 可选)：位置编码参数。
  - 数据类型：`float16`、`bfloat16`（需与 `query` 类型满足推导规则）
  - 数据格式：ND，不支持非连续。
  - Q_S > 1 时：shape 为 (B, Q_N, Q_S, KV_S) 或 (1, Q_N, Q_S, KV_S)；KV_S 非 32 对齐建议 padding 到 32 字节。
  - Q_S = 1 时：shape 为 (B, Q_N, 1, KV_S) 或 (1, Q_N, 1, KV_S)；仅支持 D 轴 16 整除。
  - 默认值：None。

#### Mask 与序列长度参数

- **atten_mask** (`Tensor`, 可选)：对 Q×K 结果做 mask，指示 Token 间相关性是否计算。
  - 数据类型：`bool`、`int8`、`uint8`
  - 数据格式：ND，不支持非连续
  - `sparse_mode`=0/1 时：支持 shape (1, Q_S, KV_S)、(B, 1, Q_S, KV_S)、(1, 1, Q_S, KV_S)；当 `input_layout` 为 BSH/BSND/BNSD/BNSD_BSND 且不传 rope 时，Q_S=1 支持 (B, KV_S)，Q_S>1 支持 (Q_S, KV_S)。
  - `sparse_mode`=2/3/4 时：shape 必须为 (2048, 2048) 或 (1, 2048, 2048) 或 (1, 1, 2048, 2048)，需用户保证为下三角。
  - 默认值：None。

- **actual_seq_lengths** (`List[int]`, 可选)：各 Batch 中 `query` 的有效序列长度。
  - 数据类型：`int64`
  - 每个 batch 有效 seqlen 不大于 `query` 中对应 batch 的 seqlen。
  - 传入长度为 1 时，所有 batch 共用；长度 ≥ batch 时取前 batch 个。
  - **TND 布局时必须传入**：每个元素值为当前 batch 与之前所有 batch 的 seqlen **累加和**（后值 ≥ 前值，非负），元素个数即 batch 值（≤ 4096）。
  - 默认值：None（表示与 `query` 的 S 相同）。

- **actual_seq_lengths_kv** (`List[int]`, 可选)：各 Batch 中 `key/value` 的有效序列长度。
  - 数据类型：`int64`
  - 约束同 `actual_seq_lengths`。
  - PageAttention 场景下必须传入。
  - 默认值：None（表示与 `key/value` 的 S 相同）。

#### int8 量化参数

- **dequant_scale1** (`Tensor`, 可选)：BMM1 后面的反量化因子，per-tensor。
  - 数据类型：`uint64`、`float32`
  - 默认值：None。

- **quant_scale1** (`Tensor`, 可选)：BMM2 前面的量化因子，per-tensor。
  - 数据类型：`float32`
  - 默认值：None。

- **dequant_scale2** (`Tensor`, 可选)：BMM2 后面的反量化因子，per-tensor。
  - 数据类型：`uint64`、`float32`
  - 默认值：None。

- **quant_scale2** (`Tensor`, 可选)：输出的量化因子，支持 per-tensor、per-channel。
  - 数据类型：`float32`、`bfloat16`（`bfloat16` 输入时两种均可，否则仅 `float32`）
  - per-channel 下：
    - 输出 BSH：所有维度乘积 = H，建议 shape (1, 1, H) 或 (H,)
    - 输出 BNSD：乘积 = Q_N × D，建议 shape (1, Q_N, 1, D) 或 (Q_N, D)
    - 输出 BSND：乘积 = Q_N × D，建议 shape (1, 1, Q_N, D) 或 (Q_N, D)
  - 默认值：None。

- **quant_offset2** (`Tensor`, 可选)：输出的量化偏移，支持 per-tensor、per-channel。
  - 若传入，类型和 shape 须与 `quant_scale2` 一致。
  - 数据类型：`float32`、`bfloat16`
  - 默认值：None（等效于 0）。

#### 伪量化参数

- **antiquant_scale** (`Tensor`, 可选)：伪量化因子，per-tensor/per-channel/per-token。
  - 数据类型：`float16`、`bfloat16`（Q_S ≥ 2 时仅 `float16`；Q_S = 1 时 per-channel 与 query 同类型，per-token 为 `float32`）
  - 建议使用 KV 伪量化分离模式（`key_antiquant_scale` / `value_antiquant_scale`）。
  - 默认值：None。

- **antiquant_offset** (`Tensor`, 可选)：伪量化偏移，per-tensor/per-channel/per-token。
  - 约束同 `antiquant_scale`（shape 须一致）。
  - 对称量化时不传（None），非对称量化时必须传入。
  - 默认值：None。

- **antiquant_mode** (`int`, 可选)：伪量化方式。0=per-channel（含 per-tensor），1=per-token。
  - Q_S ≥ 2 时无效；Q_S = 1 时传入其他值会异常。
  - 默认值：0。

#### PageAttention 参数

- **block_table** (`Tensor`, 可选)：PageAttention 中 KV 存储的 block 映射表。
  - 数据类型：`int32`
  - 数据格式：ND
  - 必须为二维：[B, maxBlockNumPerSeq]。blockid 合法性由用户保证。
  - 传入此参数即开启 PageAttention。
  - 默认值：None。

- **block_size** (`Tensor`, 可选)：PageAttention 中每个 block 最多 token 个数。
  - 数据类型：`int64`
  - Q_S > 1：最小 128，最大 512，须为 128 倍数。
  - Q_S = 1：非 0 值，最大 512。fp16/bf16 需 16 对齐，int8 需 32 对齐，推荐 128。
  - 默认值：0（不开启 PageAttention）。

#### Padding 参数

- **query_padding_size** (`Tensor`, 可选)：`query` 每个 batch 右对齐的 padding 数量。
  - 数据类型：`int64`
  - 仅 Q_S > 1 生效。
  - 需与 `actual_seq_lengths` 一起开启，否则默认为右 padding。
  - 默认值：None。

- **kv_padding_size** (`Tensor`, 可选)：`key/value` 每个 batch 右对齐的 padding 数量。
  - 数据类型：`int64`
  - 小于 0 时被置为 0。
  - 需与 `actual_seq_lengths_kv` 一起开启，否则默认为右 padding。
  - 默认值：None。

#### KV 伪量化分离参数

- **key_antiquant_scale** (`Tensor`, 可选)：KV 伪量化分离时 key 的反量化因子。
  - 数据类型：`float16`、`bfloat16`、`float32`
  - 支持 per-channel、per-tensor、per-token、per-tensor+per-head、per-token+per-head、per-token+PA 等模式。
  - 与 `value_antiquant_scale` 必须同时为空或同时非空。
  - 默认值：None。

- **key_antiquant_offset** (`Tensor`, 可选)：KV 伪量化分离时 key 的反量化偏移。
  - 约束同 `key_antiquant_scale`。
  - 与 `value_antiquant_offset` 必须同时为空或同时非空。
  - 默认值：None。

- **value_antiquant_scale** (`Tensor`, 可选)：KV 伪量化分离时 value 的反量化因子。
  - 约束同 `key_antiquant_scale`。
  - 默认值：None。

- **value_antiquant_offset** (`Tensor`, 可选)：KV 伪量化分离时 value 的反量化偏移。
  - 约束同 `key_antiquant_scale`。
  - 默认值：None。

- **key_antiquant_mode** (`int`, 可选)：key 伪量化方式。取值 0~5，详见[枚举表](#key_antiquant_mode--value_antiquant_mode-枚举说明)。默认值：0。

- **value_antiquant_mode** (`int`, 可选)：value 伪量化方式。取值 0~5。默认值：0。

#### Prefix（共享前缀）参数

- **key_shared_prefix** (`Tensor`, 可选)：Key 的共享前缀部分。
  - 数据类型：`float16`、`bfloat16`、`int8`
  - 数据格式：ND，不支持非连续。
  - shape 第一维 batch 必须为 1。
  - 与 `value_shared_prefix` 必须同时为空或同时非空。
  - 默认值：None。

- **value_shared_prefix** (`Tensor`, 可选)：Value 的共享前缀部分。
  - 约束同 `key_shared_prefix`。
  - 默认值：None。

- **actual_shared_prefix_len** (`List[int]`, 可选)：共享前缀的有效序列长度。
  - 数据类型：`int64`
  - shape 为 [1]，值不能大于 prefix 的 S。
  - 默认值：None（表示与 prefix 的 S 相同）。

#### MLA（Multi-head Latent Attention）参数

- **query_rope** (`Tensor`, 可选)：MLA 中 Query 的 RoPE 信息。
  - 数据类型：`float16`、`bfloat16`
  - 数据格式：ND，不支持非连续。
  - shape 中 D 为 64，其余维度与 `query` 一致。
  - 与 `key_rope` 必须同时配置或同时不配置。
  - 默认值：None。

- **key_rope** (`Tensor`, 可选)：MLA 中 Key 的 RoPE 信息。
  - 约束同 `query_rope`（shape 中 D 为 64，其余维度与 `key` 一致）。
  - 默认值：None。

- **key_rope_antiquant_scale** (`Tensor`, 可选)：预留参数，暂未使用。使用默认值即可。

#### 其他标量参数

- **num_heads** (`int`, 可选)：`query` 的 head 个数。BNSD 场景下需与 `query` 的 N 轴一致。默认值：1。

- **scale** (`float`, 可选)：缩放系数，通常为 **1/√D**。⚠️ 默认值为 1.0，**大多数场景需手动设置，否则计算结果将不正确**。默认值：1.0。

- **pre_tokens** (`int`, 可选)：稀疏计算中 attention 与前几个 token 的关联范围。Q_S=1 时无效。默认值：2147483647。

- **next_tokens** (`int`, 可选)：稀疏计算中 attention 与后几个 token 的关联范围。Q_S=1 时无效。默认值：2147483647。

- **num_key_value_heads** (`int`, 可选)：`key/value` 中 head 个数，用于 GQA。0 表示与 `query` 的 head 数相等。需满足 `num_heads` 整除 `num_key_value_heads`，比值 ≤ 64。BSND/BNSD/BNSD_BSND 场景需与 `key/value` 的 N 轴一致。默认值：0。

- **sparse_mode** (`int`, 可选)：稀疏模式，详见[枚举表](#sparse_mode-枚举说明)。默认值：0。

- **inner_precise** (`int`, 可选)：精度/性能模式，详见[枚举表](#inner_precise-枚举说明)。默认值：0。

- **softmax_lse_flag** (`bool`, 可选)：是否额外输出 softmax_lse（用于 Ring Attention 分布式合并）。默认值：False。

## 返回值说明

- **attention_out** (`Tensor`)：注意力输出。
  - 数据类型：`float16`、`bfloat16`、`int8`
  - 数据格式：ND
  - D 维度与 `value` 的 D 一致，其余维度与 `query` 的 shape 一致。

- **softmax_lse** (`Tensor`)：softmax 的 log-sum-exp 值（用于 Ring Attention 分布式场景合并各设备结果）。
  - `softmax_lse_flag=True` 时：
    - 一般场景：shape (B, Q_N, Q_S, 1)，dtype `float32`
    - `input_layout` 为 TND/NTD_TND 时：shape (T, Q_N, 1)，dtype `float32`
  - `softmax_lse_flag=False` 时：shape (1,)，值为 0 的 Tensor。

  Ring Attention 多设备合并公式：

  $$
  O_{global} = \frac{e^{lse_1 - lse_{max}} \cdot O_1 + e^{lse_2 - lse_{max}} \cdot O_2}{e^{lse_1 - lse_{max}} + e^{lse_2 - lse_{max}}}
  $$

  其中 $lse_{max} = max(lse_1, lse_2)$。

## 约束说明

### <a id="base_constraints"></a>通用基础约束

- 该接口支持推理场景、图模式。
- 与 PyTorch 配合使用时需 CANN 与 PyTorch 版本匹配。
- **入参为空处理**：`query` 为空则直接返回空。`query` 非空、`key/value` 为空（S2=0）则 `attention_out` 按对应 shape 返回全 0。`attention_out` 为空则返回空。
- `key` 与 `value` 的 shape 必须完全一致；非连续场景下 key/value tensorlist 中 batch 只能为 1，个数等于 query 的 B，N 和 D 需相等。
- `scale` 默认值 1.0，通常应设置为 **1/√D**，否则计算结果不正确。

### <a id="int8_constraints"></a>int8 量化约束

入参与输入/输出数据类型组合限制：

| 输入 dtype | 输出 dtype | 必须传入 | 可选传入 | 禁止传入 |
| ---------- | ---------- | -------- | -------- | -------- |
| int8 | int8 | `dequant_scale1, quant_scale1, dequant_scale2, quant_scale2` | `quant_offset2` | — |
| int8 | float16 | `dequant_scale1, quant_scale1, dequant_scale2` | — | `quant_offset2, quant_scale2` |
| float16/bf16 | int8 | `quant_scale2` | `quant_offset2` | `dequant_scale1, quant_scale1, dequant_scale2` |

- `quant_scale2` 和 `quant_offset2` 支持 per-tensor/per-channel，dtype 支持 float32/bf16。`quant_offset2` 传入时须与 `quant_scale2` 类型和 shape 一致。
- 输出 int8，`quant_scale2`/`quant_offset2` per-channel 时：不支持左 padding、Ring Attention 或 D 非 32 字节对齐。
- 输出 int8 时暂不支持 sparse=band 且 preTokens/nextTokens 为负。
- 输出为 int8 且 `quant_offset2` 非 None 时，若矩阵存在不参与计算的行，会触发拦截（建议外部做后量化）。
- `input_layout` 仅支持 BSH、BNSD、BSND、BNSD_BSND。

### <a id="pseudo_quant_constraints"></a>伪量化约束（antiquant_scale / antiquant_offset）

- 仅支持 kv dtype 为 int8。
- per-channel：shape 为 (2, KV_N, 1, D) / (2, KV_N, D) / (2, H)，dtype 与 query 同。`antiquant_mode`=0。
- per-tensor：shape 为 (2,)，dtype 与 query 同。`antiquant_mode`=0。
- per-token：shape 为 (2, B, KV_S)，dtype 固定 float32。`antiquant_mode`=1。
- 对称量化：`antiquant_offset`=None；非对称量化：两者必须同时存在。
- 算子根据 shape dim 自动判断模式（dim=1 → per-tensor，否则 per-channel）。
- **建议使用 KV 伪量化分离模式**（`key_antiquant_scale`/`value_antiquant_scale` 代替 `antiquant_scale`/`antiquant_offset`）。

### <a id="pa_constraints"></a>PageAttention 约束

**开启条件**：`block_table` 存在且有效。开启后 `key/value` 的 `input_layout` 参数无效，`key/value` 按 `block_table` 索引在连续内存中排布。

| 约束项 | Q_S > 1 | Q_S = 1 |
| ------ | ------- | ------- |
| block_size | 最小 128，最大 512，须 128 倍数 | 非 0，最大 512，推荐 128 |
| kv dtype | float16、bf16 | float16、bf16、int8 |
| query int8 | 不支持 | 不支持 |
| kv cache 排布 | BSH/BSND 仅 BnBsH；BNSD/TND 支持 BnBsH 和 BnNBsD | 同左 |
| 必须传入 actual_seq_lengths_kv | ✓ | ✓ |
| block_table 二维 | [B, ≥maxBlockNumPerSeq] | [B, ≥maxBlockNumPerSeq] |
| blocknum 下限 | ≥ 各 batch block 数之和 | 同左 |
| 支持 tensorlist | ✗ | ✗ |
| 支持左 padding | ✗ | ✗ |
| 支持伪量化 | ✗ | ✓（int8 kv） |
| 性能提示 | BnNBsD 优于 BnBsH | BnNBsD 优于 BnBsH |
| H 超 65535 时 | 开 GQA 或用 BnNBsD | 开 GQA 或用 BnNBsD |

- PageAttention 开启时，以下场景需 KV_S ≥ maxBlockNumPerSeq × block_size：
  - 传入 `atten_mask`（如 shape (B, 1, Q_S, KV_S)）
  - 传入 `pse_shift`（如 shape (B, Q_N, Q_S, KV_S)）
  - Q_S = 1 时开启 `atten_mask`（`sparse_mode` 非 2/3/4）：`atten_mask` 最后一维 ≥ block_table 第二维 × block_size
  - Q_S = 1 时开启 `pse_shift`：`pse_shift` 最后一维 ≥ block_table 第二维 × block_size
  - Q_S = 1 伪量化 per-token 模式：`antiquant_scale`/`antiquant_offset` shape 的 S ≥ 该值

### <a id="mla_constraints"></a>MLA 约束

`query_rope` 和 `key_rope` 输入时即进入 MLA 场景。

- `query_rope` 和 `key_rope` 必须同时配置或同时不配置。
- `query_rope` 的 dtype/format 与 `query` 一致；`key_rope` 的 dtype/format 与 `key` 一致。
- query 的 D 仅支持 512 或 128。

**D = 512 时**：

- sparse 支持 0/3/4。
- `query_rope`：要求 query 的 N 为 1/2/4/8/16/32/64/128，shape 中 D=64，其余维度与 query 一致。
- `key_rope`：要求 key 的 N=1、D=512，shape 中 D=64，其余维度与 key 一致。
- key/value/key_rope 支持 ND 或 NZ。NZ 格式：fp16/bf16 → [blockNum, KV_N, D/16, blockSize, 16]；int8 → [blockNum, KV_N, D/32, blockSize, 32]。
- input_layout：BSH、BSND、BNSD、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND、TND_NTD。
- 支持 PageAttention（block_size：16 的倍数、≤1024）
- 不支持：softmax_lse、左 padding、tensorlist、pse、prefix、伪量化、全量化、后量化。

**D = 128 时**：

- input_layout：BSH、BSND、TND、BNSD、NTD、BSH_BNSD、BSND_BNSD、BNSD_BSND、NTD_TND。
- `query_rope`/`key_rope` shape 中 D=64，其余维度不变。
- 不支持：左 padding、tensorlist、pse、prefix、伪量化、全量化、后量化。
- 其余约束同 TND/NTD_TND 场景。

### <a id="prefix_constraints"></a>Prefix（共享前缀）约束

- `key_shared_prefix` 和 `value_shared_prefix` 必须同时为空或同时非空。
- 非空时：两者与 `key`/`value` 维度相同、dtype 一致。
- `key_shared_prefix` 的 shape 第一维 batch=1；BNSD/BSND 下 N、D 轴与 key 一致；BSH 下 H 与 key 一致。`value_shared_prefix` 同理。两者的 S 应相等。
- `actual_shared_prefix_len` 存在时：shape=[1]，值 ≤ prefix 的 S。
- prefix 的 S + key/value 的 S ≤ 原 key/value 的 S 限制。
- 不支持：PageAttention、左 padding、tensorlist、qkv 全 int8。
- sparse=0/1 且传入 atten_mask 时：S2 ≥ `actual_shared_prefix_len` + key 的 S。

### <a id="padding_constraints"></a>Padding 约束

**query 左 padding（仅 Q_S > 1）**：

- 搬运起点 = Q_S − query_padding_size − actual_seq_lengths（须 ≥ 0）。
- 搬运终点 = Q_S − query_padding_size（须 ≤ Q_S）。
- `kv_padding_size` < 0 时被置为 0。
- 需与 `actual_seq_lengths` 一起开启。
- 不支持 PageAttention。

**kv 左 padding**：

- 搬运起点 = KV_S − kv_padding_size − actual_seq_lengths_kv（Q_S > 1 时须 ≥ 0；Q_S = 1 时若 < 0 则返回全 0）。
- 搬运终点 = KV_S − kv_padding_size（Q_S > 1 时须 ≤ KV_S；Q_S = 1 时若 < 0 则返回全 0）。
- 需与 `actual_seq_lengths_kv` 一起开启。
- 不支持 PageAttention。
- <term>Atlas A2/A3</term>：不支持 Q 为 bf16/fp16、KV 为 int4 的场景。

### <a id="tnd_constraints"></a>TND 布局约束

`input_layout` 为 TND、TND_NTD、NTD_TND 时的综合限制：

- `actual_seq_lengths` 和 `actual_seq_lengths_kv` 必须传入，元素个数即 batch 值（≤ 4096）。每个元素值为累加和（后值 ≥ 前值）。

- **query D = 512 时**：

  - sparse：0/3/4。
  - 支持 TND、TND_NTD。
  - 支持 PageAttention（`actual_seq_lengths_kv` 长度 = kv batch 数，值 ≤ KV_S）。
  - query N = 1/2/4/8/16/32/64/128，kv N = 1。
  - `query_rope` 和 `key_rope` 非空（D=64）。
  - 不支持：左 padding、tensorlist、pse、prefix、伪量化、全量化。
- **query D ≠ 512 时**：
  - 无 rope：Q_D、K_D、V_D = 128 或 Q_D、K_D = 192 且 V_D = 128/192（NTD 不支持 V_D=192；NTD_TND 要求 Q_D、K_D = 128/192，V_D = 128）。
  - 有 rope：Q_D、K_D、V_D = 128。
  - 支持 TND、NTD、NTD_TND。
  - PageAttention 仅 block_size=16 对齐且 ≤ 1024。
  - 不支持：左 padding、tensorlist、pse、prefix、伪量化、全量化、后量化。

### <a id="qs_gt_1"></a>Q_S > 1（Prefill）专属约束

- **B 轴**：≤ 65536。D 轴非 32 字节对齐时 ≤ 128。
- **N 轴**：≤ 256。
- **D 轴**：≤ 512。BSH/BSND 时建议 N×D < 65535。
- **S 轴**：≤ 20971520 (20M)。
- **D 对齐**：int8 → 32 对齐，int4 → 64 对齐，fp16/bf16 → 16 对齐。
- **长序列超时风险**：B×N×S×D 过大时可能超时（aicore timeout/trap error），建议 S 切分。典型高风险场景：
  - B=1, Q_N=20, Q_S=2097152, D=256, KV_N=1, KV_S=2097152
  - B=1, Q_N=2, Q_S=20971520, D=256, KV_N=2, KV_S=20971520
  - B=20, Q_N=1, Q_S=2097152, D=256, KV_N=1, KV_S=2097152
  - B=1, Q_N=10, Q_S=2097152, D=512, KV_N=1, KV_S=2097152
- **sparse_mode**：仅支持 0/1/2/3/4，其他值报错。
  - 0：`atten_mask`=None 或在左 padding 时忽略 `pre_tokens`/`next_tokens`。
  - 2/3/4：mask shape 必须为 (S,S) 或 (1,S,S) 或 (1,1,S,S)，其中 S=2048，且用户保证下三角。
  - 1/2/3：忽略 `pre_tokens`/`next_tokens`。
- **kvCache 反量化合成参数**：仅支持 int8 → float16。key/value 的 data range × antiquant_scale 的 data range 须在 (−1, 1) 内（高性能模式）否则需高精度模式。
- **pse_shift**：query dtype 为 fp16/bf16/int8 时可用。fp16 query + pse_shift → 强制高精度模式。Q_S ≥ query S，KV_S ≥ key S。prefix 时 KV_S ≥ actual_shared_prefix_len + key S。
- **GQA 伪量化 + KV NZ 格式**：
  - query: bf16, key/value: int8, D=128, Q_S=1~16。
  - input_layout: BSH/BSND/BNSD。
  - 仅 PageAttention（block_size=128 或 512）。
  - key/value NZ 格式：[blockNum, KV_N, D/32, blockSize, 32]。
  - key/value_antiquant_scale dtype：per-channel → bf16, per-token → float32。
  - 仅 KV 分离模式，不支持 offset、rope、左 padding、tensorlist、pse、prefix、后量化。
  - num_heads / num_key_value_heads 支持组合：(10,1)、(64,8)、(80,8)、(128,16)。
  - MTP=0：sparse_mode=0 不传 mask。MTP>0 且 <16：sparse_mode=3 传优化 mask (2048×2048)。
  - 仅高性能模式。

### <a id="qs_eq_1"></a>Q_S = 1（Decode）专属约束

- **B 轴**：≤ 65536。
- **N 轴**：≤ 256。
- **D 轴**：≤ 512。
- **S 轴**（KV）：≤ 262144。
- **qkv 全 int8**：不支持。
- **int4（int32）伪量化**：
  - PyTorch 入图调用仅支持 KV int4 拼接为 int32 输入（建议通过 dynamic_quant 生成：1 个 int32 包含 8 个 int4）。
  - KV N、D 或 H 为实际值的 1/8（prefix 同理）。
  - int4 仅支持 D 64 对齐（int32 支持 D 8 对齐）。
- **actual_seq_lengths**：非 TND 布局时该参数无效。
- **actual_seq_lengths_kv**：非 TND 布局时传入长度为 1/≥batch；传入值 ≤ 对应 KV_S。
- **pse_shift**：dtype 与 query 一致，仅支持 D 轴 16 整除。
- **kv 左 padding**：搬运起点/终点 < 0 时返回全 0；需与 `actual_seq_lengths_kv` 一起开启。
- **int4 kv**：不支持左 padding、page attention（Q: bf16/fp16 + KV: int4）。

## 调用示例

### 示例 1：MHA + Prefill（基础场景）

最简单的全量计算调用：单 batch、8 head、D=128、BNSD 布局。

```python
import torch
import torch_npu
import math

# ===== 数据准备 =====
B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

# query/key/value 形状: (B, N, S, D) — BNSD 布局
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# scale 必须设置为 1/sqrt(D)
scale = 1.0 / math.sqrt(D)

# actual_seq_lengths: 长度为 1 表示所有 batch 共用
# actual_seq_lengths_kv: key/value 的有效长度
act_seq_len = [Q_S]
act_seq_len_kv = [KV_S]

# ===== 调用算子 =====
# MHA: num_heads == num_key_value_heads (或不传 num_key_value_heads 默认为 0)
out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    actual_seq_lengths=act_seq_len,
    actual_seq_lengths_kv=act_seq_len_kv,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> **关键参数说明**：

> - `scale=1/√128`：必设，控制 softmax 之前 QK^T 的缩放。设错会导致 attention 分布异常。
> - `pre_tokens=65535, next_tokens=65535`：表示不限制 attention 范围（等价于双向 attention），对应的约束见 [sparse_mode 枚举说明](#sparse_mode-枚举说明)。
> - `input_layout="BNSD"`：表示输入为 (B,N,S,D)，见 [input_layout 枚举说明](#input_layout-枚举说明)。
> - 此场景命中 Q_S > 1，走 PromptFlashAttention 分支，约束见 [Q_S > 1 约束](#qs_gt_1)。

### 示例 2：GQA + Decode + PageAttention

分组查询注意力 + KV Cache + PageAttention 的增量推理场景。

```python
import torch
import torch_npu
import math

# ===== 数据准备 =====
B = 1
Q_N = 8          # query heads
KV_N = 2         # key/value heads (GQA: Q_N / KV_N = 4)
Q_S = 1          # decode 阶段 query 长度 = 1
D = 128
KV_S = 2048      # KV Cache 总长度
block_size = 128
# 计算需要的 block 数量
max_block_num = (KV_S + block_size - 1) // block_size  # = 16

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
# KV Cache: (blocknum, blocksize, H) — BnBsH 格式
# H = KV_N * D = 2 * 128 = 256
k = torch.randn(max_block_num * B, block_size, KV_N * D, dtype=torch.float16).npu()
v = torch.randn(max_block_num * B, block_size, KV_N * D, dtype=torch.float16).npu()

# block_table: (B, max_block_num)，存储 block id
block_table = torch.arange(max_block_num, dtype=torch.int32).unsqueeze(0).npu()

scale = 1.0 / math.sqrt(D)

# PageAttention 场景必须传入 actual_seq_lengths_kv
act_seq_len_kv = [KV_S]

# ===== 调用算子 =====
out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    actual_seq_lengths_kv=act_seq_len_kv,
    num_heads=Q_N,
    num_key_value_heads=KV_N,   # GQA: 指定 KV head 数
    input_layout="BSH",          # kv cache 为 BnBsH 时用 BSH
    scale=scale,
    block_table=block_table,
    block_size=block_size
)

print(out.shape)  # torch.Size([1, 8, 1, 128])
```

> **关键参数说明**：

> - `num_key_value_heads=2`：启用 GQA，num_heads / num_key_value_heads = 4。该参数不传时默认为 0（即 MHA），见 [参数说明](#详细参数说明)。
> - `block_table` + `block_size`：传入即开启 PageAttention，kv cache 排布见 [PageAttention 约束](#pa_constraints)。
> - `input_layout="BSH"`：kv cache 为 BnBsH 格式时必须用 BSH。若 kv cache 为 BnNBsD 格式且 query 为 BNSD，则可用 BNSD。
> - `actual_seq_lengths_kv`：PageAttention 场景必须传入，对应约束见 [PageAttention 约束](#pa_constraints)。
> - 此场景 Q_S=1，走 IncreFlashAttention 分支，约束见 [Q_S = 1 约束](#qs_eq_1)。

### 示例 3：MHA + Causal Mask（带 atten_mask）

使用 causal attention mask 的标准 decoder 场景。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 164

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# 下三角 causal mask: (1, Q_S, KV_S)
mask = torch.tril(torch.ones(Q_S, KV_S, dtype=torch.bool)).unsqueeze(0).npu()

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    atten_mask=mask,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    sparse_mode=0,     # defaultMask: 使用完整 mask 矩阵
    pre_tokens=65535,
    next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> **关键参数说明**：
> - `atten_mask`：shape (1, Q_S, KV_S)，bool 类型。更多 mask shape 选项见 [atten_mask 参数说明](#详细参数说明)。
> - `sparse_mode=0`：defaultMask 模式，使用 `pre_tokens`/`next_tokens` 控制计算范围。此处为完整双向 attention。
> - 也可使用 `sparse_mode=3`（rightDownCausal）+ 2048×2048 优化 mask。各模式含义见 [sparse_mode 枚举说明](#sparse_mode-枚举说明)。

### 示例 4：MLA + Decode（Multi-head Latent Attention）

DeepSeek 类模型的 MLA 推理场景。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 16, 1, 512   # MLA 场景 D=512
KV_N, KV_S = 1, 2048
rope_D = 64                       # RoPE 维度固定 64

# query: (B, Q_N, Q_S, D)
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
# key/value: (B, KV_N, KV_S, D)
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# RoPE 张量: shape 中 D=64，其余维度与 q/k 一致
q_rope = torch.randn(B, Q_N, Q_S, rope_D, dtype=torch.float16).npu()
k_rope = torch.randn(B, KV_N, KV_S, rope_D, dtype=torch.float16).npu()

scale = 1.0 / math.sqrt(D + rope_D)  # MLA 场景有效维度 = D + 64

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    query_rope=q_rope,
    key_rope=k_rope,
    num_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="BNSD",
    scale=scale,
    sparse_mode=0
)

print(out.shape)  # torch.Size([1, 16, 1, 512])
```

> **关键参数说明**：

> - `query_rope` + `key_rope`：传入即进入 MLA 分支，必须同时传入。RoPE 维度固定 64。二者 dtype/format 须与 query/key 一致。
> - D=512：MLA prefill 场景的 D 为 512 或 128。D=512 时可开启 PageAttention、支持 NZ 格式等。约束详见 [MLA 约束](#mla_constraints)。
> - 此场景不支持：softmax_lse、左 padding、pse、prefix、伪量化、全量化。

### 示例 5：int8 后量化

仅对输出做 int8 量化（post-training quantization），输入保持 fp16。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# pertensor 量化因子: shape (1, 1, 1) 或标量 shape
quant_scale2 = torch.tensor([0.01], dtype=torch.float32).npu()
# 不传 quant_offset2 则默认为 0

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    quant_scale2=quant_scale2,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.dtype)  # torch.int8
print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> **关键参数说明**：

> - `quant_scale2`：传入即触发后量化（输出 int8）。不传 `quant_offset2` 则偏移为 0。
> - int8 输出场景下，**禁止**同时传入 `dequant_scale1`/`quant_scale1`/`dequant_scale2`。约束详见 [int8 量化约束](#int8_constraints)。
> - 如需 per-channel 量化，shape 建议 (Q_N, D)。详见 [quant_scale2 参数说明](#详细参数说明)。

### 示例 6：int8 全量化

输入 int8 → 内部反量化 → fp16 计算 → 量化输出 int8。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

# int8 输入（需确保 D 轴 32 对齐）
q = torch.randint(-128, 127, (B, Q_N, Q_S, D), dtype=torch.int8).npu()
k = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()

# BMM1 后反量化因子 (per-tensor)
deq_scale1 = torch.tensor([0.01], dtype=torch.float32).npu()
# BMM2 前量化因子
quant_scale1 = torch.tensor([0.02], dtype=torch.float32).npu()
# BMM2 后反量化因子
deq_scale2 = torch.tensor([0.03], dtype=torch.float32).npu()
# 输出量化因子
quant_scale2 = torch.tensor([0.04], dtype=torch.float32).npu()

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    dequant_scale1=deq_scale1,
    quant_scale1=quant_scale1,
    dequant_scale2=deq_scale2,
    quant_scale2=quant_scale2,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.dtype)  # torch.int8
```

> **关键参数说明**：

> - 四个量化参数必须**同时存在**。`quant_offset2` 可选，不传时偏移为 0。
> - D 轴需 32 对齐。input_layout 仅支持 BSH/BNSD/BSND/BNSD_BSND。
> - 约束详见 [int8 量化约束](#int8_constraints)。

### 示例 7：伪量化（KV 分离模式）+ Decode

key/value 为 int8，通过 per-channel 反量化因子转为 fp16 参与计算。使用推荐的 KV 分离模式。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 1, 128
KV_N, KV_S = 2, 2048

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.bfloat16).npu()
k = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()

# KV 分离 per-channel 量化因子: shape (1, KV_N, 1, D) 或 (1, KV_N, D) 或 (1, H)
key_antiquant_scale = torch.ones(1, KV_N, 1, D, dtype=torch.bfloat16).npu() * 0.01
value_antiquant_scale = torch.ones(1, KV_N, 1, D, dtype=torch.bfloat16).npu() * 0.01
# 对称量化：不传 offset

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    key_antiquant_scale=key_antiquant_scale,
    value_antiquant_scale=value_antiquant_scale,
    key_antiquant_mode=0,     # per-channel
    value_antiquant_mode=0,
    num_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="BNSD",
    scale=scale
)

print(out.dtype)   # torch.bfloat16
print(out.shape)   # torch.Size([1, 8, 1, 128])
```

> **关键参数说明**：

> - 使用 KV 分离模式（优于 `antiquant_scale`/`antiquant_offset` 模式）。
> - `key_antiquant_mode=0`：per-channel 模式。此处 Q_S=1 + query bf16 + kv int8 + key/value_antiquant_scale bf16 → 满足 [Q_S = 1 约束](#qs_eq_1) 中的 kv 伪量化参数分离条件。
> - 当 `key_antiquant_scale`/`value_antiquant_scale` 与 `antiquant_scale` 同时传入时，以 KV 分离参数为准。
> - 各模式的 shape 和 dtype 要求见 [key_antiquant_mode 枚举表](#key_antiquant_mode--value_antiquant_mode-枚举说明)。

### 示例 8：Prefix + Prefill（共享前缀）

系统提示词/共享前缀场景：prefix 部分作为公共 KV，与每次请求的 KV 拼接计算。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024
prefix_S = 32

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# 共享前缀: batch=1, S=prefix_S, N/D 与 k/v 一致
key_prefix = torch.randn(1, KV_N, prefix_S, D, dtype=torch.float16).npu()
value_prefix = torch.randn(1, KV_N, prefix_S, D, dtype=torch.float16).npu()

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    key_shared_prefix=key_prefix,
    value_shared_prefix=value_prefix,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> **关键参数说明**：

> - `key_shared_prefix` + `value_shared_prefix`：必须同时传入。shape 第一维 batch=1。约束见 [Prefix 约束](#prefix_constraints)。
> - 如果 S2（KV_S + prefix_S）超出原 KV_S 限制则报错。若传入 `atten_mask`，S2 需大于 prefix_S + KV_S。
> - 不支持 PageAttention、左 padding、tensorlist、qkv 全 int8。

### 示例 9：图模式（torch.compile + torchair）

通过 `torch.compile` + `torchair` backend 将算子纳入计算图。

```python
import torch
import torch_npu
import math
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

# ===== 图模式配置 =====
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)

# ===== 数据准备 =====
B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
scale = 1.0 / math.sqrt(D)


class Model(torch.nn.Module):
    def forward(self):
        return torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=Q_N,
            input_layout="BNSD",
            scale=scale,
            pre_tokens=65535,
            next_tokens=65535
        )


# ===== 编译 + 执行 =====
with torch.no_grad():
    model = Model()
    compiled = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    graph_out = compiled()

    # 与单算子结果对比
    single_out, _ = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=Q_N,
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535
    )
    print("graph output shape:", graph_out[0].shape)
    print("single op output shape:", single_out.shape)
    print("match:", torch.allclose(graph_out[0], single_out, atol=1e-2))
```

> **说明**：图模式下参数含义与单算子模式一致。需确保 `torchair` 与 CANN 版本匹配。

## 参考资源

- **BMM1（Batch Matrix Multiply 1）**：第一次矩阵乘法，计算 attention score：$attention\_score = query \cdot key^{\top}$。
- **BMM2（Batch Matrix Multiply 2）**：第二次矩阵乘法，将 attention 权重应用到 value：$attention\_out = atten\_prob \cdot value$，其中 $atten\_prob = softmax(\cdot)$。
- **Ring Attention**：分布式注意力算法，将长序列按 block 切分到多设备，通过环形通信逐块计算注意力，降低单卡显存占用。单卡退化为标准注意力。多设备合并依赖 `softmax_lse_flag=True` 输出的 LSE 值。详见[返回值说明](#返回值说明)。
