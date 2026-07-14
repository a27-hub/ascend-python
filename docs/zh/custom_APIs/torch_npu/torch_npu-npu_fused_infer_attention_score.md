# torch_npu.npu_fused_infer_attention_score

> [!WARNING]
> 该接口后续版本会废弃，请使用最新接口。对应 CANN 算子 [aclnnFusedInferAttentionScoreV5](../../../../aclnnFusedInferAttentionScoreV5.md)。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    | √  |

> [!NOTE]
> 本接口在 torch_npu 侧对 Atlas A3 训练/推理系列产品的支持范围可能超出对应 CANN 算子接口。具体约束以本文档为准。

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

- API功能：适配增量&全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。当`query`矩阵的S为1，进入IncreFlashAttention分支，其余场景进入PromptFlashAttention分支。支持的两种场景如下。
  - PromptFlashAttention：用于prefill/prompt阶段，通常`query/key/value`都是整段序列的张量，可配合`atten_mask`和可选`pse_shift`使用。该模式更关注吞吐，长序列场景下需要关注超时风险与切分策略。
  - IncreFlashAttention：用于decode阶段，`query`的S维为1，`key/value`通常来自KV Cache。该模式更关注单步延迟与KV Cache访问效率，启用page attention可以改善吞吐/访存，但会引入额外约束。

  **说明：**

  decode场景下特有KV Cache：KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（KV Cache）。

- 计算公式：

  基本公式：

  $$
  attention\_out = softmax \left(scale * (query * key^\top) + atten\_mask \right) * value
  $$

  位置编码：

  $$
  attention\_out = softmax\left( scale \cdot (query \cdot key^\top) + pse\_shift + atten\_mask \right) \cdot value
  $$

  全量化公式：

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
  score_{fp} = scale \cdot (query_{fp} \cdot key_{fp}^\top) + mask
  $$

  $$
  score_{int8} = quantize(score_{fp}, quant\_scale1)
  $$

  $$
  atten\_prob_{int8} = softmax(score_{int8})
  $$

  $$
  out_{fp} = atten\_prob_{int8} \cdot value_{fp}
  $$

  $$
  attention\_out = quantize(out_{fp}, quant\_scale2, quant\_offset2)
  $$

  后量化公式：

  $$
  out_{fp} = softmax(scale \cdot (query \cdot key^\top) + mask) \cdot value
  $$

  $$
  attention\_out = quantize(out_{fp}, quant\_scale2, quant\_offset2)
  $$
  其中，`quantize`为量化函数。

  伪量化公式：

  $$
  key_{fp} = key \cdot antiquant\_scale + antiquant\_offset
  $$

  $$
  value_{fp} = value \cdot antiquant\_scale + antiquant\_offset
  $$

  $$
  attention\_out = softmax(scale \cdot (query \cdot key_{fp}^\top) + mask) \cdot value_{fp}
  $$

  MLA场景：

  $$
  query_{rot} = RoPE(query, query\_rope)
  $$

  $$
  key_{rot} = RoPE(key, key\_rope)
  $$

  $$
  attention\_out = softmax(scale \cdot (query_{rot} \cdot key_{rot}^\top) + mask) \cdot value
  $$
  其中，`RoPE`为旋转位置编码函数。

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```python
torch_npu.npu_fused_infer_attention_score(query, key, value, *, pse_shift=None, atten_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, block_table=None, query_padding_size=None, kv_padding_size=None, key_antiquant_scale=None, key_antiquant_offset=None, value_antiquant_scale=None, value_antiquant_offset=None, key_shared_prefix=None, value_shared_prefix=None, actual_shared_prefix_len=None, query_rope=None, key_rope=None, key_rope_antiquant_scale=None, num_heads=1, scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", num_key_value_heads=0, sparse_mode=0, inner_precise=0, block_size=0, antiquant_mode=0, softmax_lse_flag=False, key_antiquant_mode=0, value_antiquant_mode=0) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

> [!NOTE]
>
> - query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示隐藏层的大小、N（Head Num）表示多头数、D（Head Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - Q_S和S1表示query shape中的S，KV_S和S2表示key和value shape中的S，Q_N表示num\_query\_heads，KV_N表示num\_key\_value\_heads。

- **query** (`Tensor`)：必选参数。attention结构的Query输入，不支持非连续的Tensor，数据类型支持`float16`、`bfloat16`、`int8`，数据格式支持$ND$。

- **key** (`Tensor`)：必选参数。attention结构的Key输入，不支持非连续的Tensor，数据类型支持`float16`、`bfloat16`、`int8`、`int4`（`int32`），数据格式支持$ND$。

- **value** (`Tensor`)：必选参数。attention结构的Value输入，不支持非连续的Tensor，数据类型支持`float16`、`bfloat16`、`int8`、`int4`（`int32`），数据格式支持$ND$。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **pse_shift** (`Tensor`)：可选参数。在attention结构内部的位置编码参数，数据类型支持`float16`、`bfloat16`，数据类型与`query`的数据类型需满足数据类型推导规则。不支持非连续的Tensor，数据格式支持$ND$。如不使用该功能时可传入None。
    - Q_S大于1，要求在`pse_shift`为`float16`类型时，此时的`query`为`float16`或`int8`类型；而在`pse_shift`为`bfloat16`类型时，要求此时的`query`为`bfloat16`类型。输入shape类型需为(B, Q\_N, Q_S, KV_S)或(1, Q\_N, Q_S, KV_S)，其中Q_S为`query`的shape中的S，KV_S为`key`和`value`的shape中的S。对于`pse_shift`的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
    - Q_S为1，要求在`pse_shift`为`float16`类型时，此时的`query`为`float16`类型；而在`pse_shift`为`bfloat16`类型时，要求此时的`query`为`bfloat16`类型。输入shape类型需为(B, Q\_N, 1, KV_S)或(1, Q\_N, 1, KV_S)，KV_S为`key`和`value`的shape中的S。对于`pse_shift`的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。

- **atten_mask** (`Tensor`)：可选参数。对Q（`query`）、K（`key`）的结果进行mask，用于指示是否计算Token间的相关性，数据类型支持`bool`、`int8`和`uint8`。不支持非连续的Tensor，数据格式支持$ND$。默认值为None。
    - `sparse_mode`为0、1时
        - 支持shape传入(1,Q_S,KV_S)、(B,1,Q_S,KV_S)、(1,1,Q_S,KV_S)。
        - 当输入`input_layout`为BSH、BSND、BNSD、BNSD_BSND时，且query、key、value的D相等，并且不传`query_rope`和`key_rope`时，Q_S为1可支持传入(B,KV_S)，Q_S大于1时可支持传入(Q_S,KV_S)。
        - 如果Q\_S、KV\_S非16或32对齐，可以取到向上对齐的值。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
    - `sparse_mode`为2、3、4时，shape输入支持(2048,2048)或(1,2048,2048)或(1,1,2048,2048)。

- **actual_seq_lengths** (`List[int]`)：可选参数。代表不同Batch中`query`的有效seqlen，数据类型支持`int64`。默认值为None，表示和`query`的shape的s长度相同。

    限制：该入参中每个Batch的有效seqlen应该不大于`query`中对应Batch的seqlen。seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度大于等于Batch时取seqlen的前Batch个数。其他长度不支持。当`query`的input\_layout为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。

- **actual_seq_lengths_kv** (`List[int]`)：可选参数。代表不同Batch中`key`/`value`的有效seqlenKv，数据类型支持`int64`。默认值为None，表示和`key`/`value`的shape的S长度相同。
    限制：该入参中每个batch的有效Sequence Length应该不大于`key`/`value`中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。不同Q\_S值有不同的约束，具体参见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- **dequant_scale1** (`Tensor`)：可选参数。数据类型支持`uint64`、`float32`。数据格式支持$ND$，表示BMM1后面的反量化因子，支持pertensor。如不使用该功能时传入None。BMM1定义请见[参考资源](#zh-cn_topic_0000001832267082_section28169228374)。
- **quant_scale1** (`Tensor`)：可选参数。数据类型支持`float32`。数据格式支持$ND$，表示BMM2前面的量化因子，支持pertensor。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。BMM2定义请见[参考资源](#zh-cn_topic_0000001832267082_section28169228374)。
- **dequant_scale2** (`Tensor`)：可选参数。数据类型支持`uint64`、`float32`。数据格式支持$ND$，表示BMM2后面的反量化因子，支持pertensor。如不使用该功能时传入None。
- **quant_scale2** (`Tensor`)：可选参数。数据类型支持`float32`、`bfloat16`。数据格式支持$ND$，表示输出的量化因子，支持pertensor、perchannel。当输入为`bfloat16`时，同时支持`float32`和`bfloat16`，否则仅支持`float32`。perchannel格式，当输出layout为BSH时，要求`quant_scale2`所有维度的乘积等于H；其他layout要求乘积等于Q\_N\*D（建议输出layout为BSH时，`quant_scale2` shape传入\(1, 1, H\)或\(H,\)；输出为BNSD时，建议传入\(1, Q\_N, 1, D\)或\(Q\_N, D\)；输出为BSND时，建议传入\(1, 1, Q\_N, D\)或\(Q\_N, D\)）。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
- **quant_offset2** (`Tensor`)：可选参数。数据类型支持`float32`、`bfloat16`。数据格式支持ND，表示输出的量化偏移，支持pertensor、perchannel。若传入`quant_offset2`，需保证其类型和shape信息与`quant_scale2`一致。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
- **antiquant_scale** (`Tensor`)：可选参数。数据类型支持`float16`、`bfloat16`、`float32`。数据格式支持$ND$，表示伪量化因子，支持pertensor、perchannel、pertoken。Q\_S为1时只支持perchannel，Q\_S大于等于2时只支持`float16`和`float32`（`float32`仅PageAttention场景下支持）。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
- **antiquant_offset** (`Tensor`)：可选参数。数据类型支持`float16`、`bfloat16`、`float32`。数据格式支持$ND$，表示伪量化偏移，支持pertensor、perchannel、pertoken。Q\_S为1时只支持perchannel，Q\_S大于等于2时只支持`float16`和`float32`（`float32`仅PageAttention场景下支持）。使用时，shape必须与`antiquant_scale`保持一致。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。建议使用KV伪量化参数分离模式。
- **block_table** (`Tensor`)：可选参数。数据类型支持`int32`。数据格式支持$ND$。表示page attention中KV存储使用的block映射表，如不使用该功能可传入None。
- **query_padding_size** (`Tensor`)：可选参数。数据类型支持`int64`。数据格式支持ND。表示`query`中每个batch的数据是否右对齐，且右对齐的个数是多少。仅支持Q\_S大于1，其余场景该参数无效。默认值为None。
- **kv_padding_size** (`Tensor`)：可选参数。数据类型支持`int64`。数据格式支持$ND$。表示`key`、`value`中每个batch的数据是否右对齐，且右对齐的个数是多少。默认值为None。
- **key_antiquant_scale** (`Tensor`)：可选参数。数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$，kv伪量化参数分离时表示key的反量化因子。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。通常支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理scale、pertoken叠加per head并使用page attention模式管理scale。

- **key_antiquant_offset** (`Tensor`)：可选参数。数据类型支持`float16`、`bfloat16`、`float32`。数据格式支持$ND$，kv伪量化参数分离时表示`key`的反量化偏移。支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理offset、pertoken叠加per head并使用page attention模式管理offset。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
- **value_antiquant_scale** (`Tensor`)：可选参数。数据类型支持`float16`、`bfloat16`、`float32`。数据格式支持$ND$，kv伪量化参数分离时表示`value`的反量化因子。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。通常支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理scale、pertoken叠加per head并使用page attention模式管理scale。

- **value_antiquant_offset** (`Tensor`)：可选参数。数据类型支持`float16`、`bfloat16`、`float32`。数据格式支持$ND$，kv伪量化参数分离时表示`value`的反量化偏移，支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理offset、pertoken叠加per head并使用page attention模式管理offset。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
- **key_shared_prefix** (`Tensor`)：可选参数。attention结构中Key前缀部分的参数，数据类型支持`float16`、`bfloat16`、`int8`，不支持非连续的Tensor，数据格式支持$ND$。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
- **value_shared_prefix** (`Tensor`)：可选参数。attention结构中Value前缀部分的参数，数据类型支持`float16`、`bfloat16`、`int8`，不支持非连续的Tensor，数据格式支持$ND$。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
- **actual_shared_prefix_len** (`List[int]`)：可选参数。代表`key_shared_prefix`/`value_shared_prefix`的有效Sequence Length。数据类型支持`int64`。默认值为None，表示和`key_shared_prefix`/`value_shared_prefix`的s长度相同。限制：该入参中的有效Sequence Length应该不大于`key_shared_prefix`/`value_shared_prefix`中的Sequence Length。
- **query_rope** (`Tensor`)：可选参数。表示MLA（Multi-head Latent Attention）结构中的Query的rope信息，数据类型支持`float16`、`bfloat16`，不支持非连续的Tensor，数据格式支持$ND$。
- **key_rope** (`Tensor`)：可选参数。表示MLA（Multi-head Latent Attention）结构中的Key的rope信息，数据类型支持`float16`、`bfloat16`，不支持非连续的Tensor，数据格式支持$ND$。
- **key_rope_antiquant_scale** (`Tensor`)：可选参数。预留参数，暂未使用，使用默认值即可。
- **num_heads** (`int`)：可选参数。代表`query`的head个数，数据类型支持`int64`，在BNSD场景下，需要与`query`的N轴shape值相同，否则执行异常。
- **scale** (`float`)：可选参数。通常是D开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持`float`。数据类型与`query`的数据类型需满足数据类型推导规则。默认值为1.0。
- **pre_tokens** (`int`)：可选参数。用于稀疏计算，表示attention需要和前几个Token计算关联，数据类型支持`int64`。默认值为2147483647，Q\_S为1时该参数无效。
- **next_tokens** (`int`)：可选参数。用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`int64`。默认值为2147483647，Q\_S为1时该参数无效。
- **input_layout** (`string`)：可选参数。用于标识输入`query`、`key`、`value`的数据排布格式，默认值为"BSH"。

    > [!NOTE]
    > 注意排布格式带下划线时，下划线左边表示输入`query`的layout，下划线右边表示输出output的格式，算子内部会进行layout转换。

    支持BSH、BSND、BNSD、BNSD\_BSND（输入为BNSD时，输出格式为BSND，仅支持Q\_S大于1）、BSH\_NBSD、BSND\_NBSD、BNSD\_NBSD（输出格式为NBSD时，仅支持Q\_S大于1且小于等于16）、TND、TND\_NTD、NTD\_TND（TND相关场景综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)）。

    **典型 shape 说明：**

    | input_layout | Q_S | query shape | key/value shape | output shape | 说明 |
    |:---|:---|:---|:---|:---|:---|
    | BSH | — | (B, S, H) | (B, S, H) | (B, S, H) | 标准格式，H = N × D |
    | BSND | — | (B, S, N, D) | (B, S, N, D) | (B, S, N, D) | 推荐使用 |
    | BNSD | >1 | (B, N, S, D) | (B, N, S, D) | (B, N, S, D) | — |
    | BNSD_BSND | >1 | (B, N, S, D) | (B, N, S, D) | (B, S, N, D) | 输入BNSD，输出BSND |
    | BSH_NBSD | >1, ≤16 | (B, S, H) | (B, S, H) | (B, N, blockSize, D) | 输出NBSD，S即blockSize |
    | BSND_NBSD | >1, ≤16 | (B, S, N, D) | (B, S, N, D) | (B, N, blockSize, D) | 输出NBSD |
    | BNSD_NBSD | >1, ≤16 | (B, N, S, D) | (B, N, S, D) | (B, N, blockSize, D) | 输出NBSD |
    | TND | — | (T, N, D) | (T, N, D) | (T, N, D) | T=∑Sᵢ，需配合`actual_seq_lengths` |
    | TND_NTD | — | (T, N, D) | (T, N, D) | (N, T, D) | 输出NTD |
    | NTD_TND | — | (N, T, D) | (N, T, D) | (T, N, D) | 输入NTD，输出TND |

    其中 TND / NTD 场景下，`actual_seq_lengths` 必须传入，且每个元素的值表示当前Batch与之前所有Batch的seqlen之和，后一个元素的值必须大于等于前一个元素的值。

- **num_key_value_heads** (`int`)：可选参数。代表`key`、`value`中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景，数据类型支持`int64`。默认值为0，表示`key`/`value`和`query`的head个数相等，需要满足`num_heads`整除`num_key_value_heads`，`num_heads`与`num_key_value_heads`的比值不能大于64。在BSND、BNSD、BNSD\_BSND（仅支持Q\_S大于1）场景下，还需要与`key`/`value`的N轴shape值相同，否则执行异常。
- **sparse_mode** (`int`)：可选参数。表示sparse的模式。数据类型支持`int64`。Q\_S为1且不带rope输入时该参数无效。input\_layout为TND、TND\_NTD、NTD\_TND时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

    **sparse_mode 枚举说明：**

    | 值 | 模式 | atten_mask 要求 | pre_tokens / next_tokens 行为 | Q_S=1 | 支持状态 |
    |:--|:-----|:----------------|:------------------------------|:------|:--------|
    | 0 | defaultMask | 可选；传入时需完整的 S1×S2 矩阵 | 未传 mask 时忽略（内部赋 INT_MAX）；传入则生效 | 无效 | 支持 |
    | 1 | allMask | 必须传入完整的 S1×S2 矩阵 | 忽略，按规则赋值 | 无效 | 支持 |
    | 2 | leftUpCausal | 必须传入优化后的 2048×2048 下三角矩阵 | 忽略，按规则赋值 | 无效 | 支持 |
    | 3 | rightDownCausal | 必须传入优化后的 2048×2048 矩阵（右顶点下三角） | 忽略，按规则赋值 | 无效 | 支持 |
    | 4 | band | 必须传入优化后的 2048×2048 矩阵 | 生效 | 无效 | 支持 |
    | 5 | prefix | — | — | — | 暂不支持 |
    | 6 | global | — | — | — | 暂不支持 |
    | 7 | dilated | — | — | — | 暂不支持 |
    | 8 | block_local | — | — | — | 暂不支持 |

    > [!NOTE]
    > Q_S > 1 时仅支持 0、1、2、3、4，取其他值会报错。sparse_mode = 0 且左 padding 场景传入 atten_mask 时，忽略 pre_tokens / next_tokens。

- **inner_precise** (`int`)：可选参数。一共4种模式：0、1、2、3。一共两位bit位，第0位（bit0）表示高精度或者高性能选择，第1位（bit1）表示是否做行无效修正。数据类型支持`int64`。Q\_S\>1时，`sparse_mode`为0或1，并传入用户自定义mask的情况下，建议开启行无效；Q\_S为1时该参数仅支持`inner_precise`为0和1。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

    | inner_precise | 模式 | 行无效修正 |
    |:--------------|:-----|:-----------|
    | 0 | 高精度 | × |
    | 1 | 高性能 | × |
    | 2 | 高精度 | √ |
    | 3 | 高性能 | √ |

    > [!NOTE]
    > `bfloat16`和`int8`不区分高精度和高性能，行无效修正对`float16`、`bfloat16`和`int8`均生效。当前0、1为保留配置值，当计算过程中"参与计算的mask部分"存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来开启行无效功能以提升精度，但是该配置会导致性能下降。如果算子可判断出存在无效行场景，会自动开启无效行计算（例如sparse_mode为3，Sq > Skv场景）。

- **block_size** (`int`)：可选参数。page attention中KV存储每个block中最大的token个数，默认为0，数据类型支持`int64`。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
- **antiquant_mode** (`int`)：可选参数。表示伪量化方式，传入0时表示为perchannel（perchannel包含pertensor），传入1时表示pertoken。默认值为0。

    Q\_S大于等于2时该参数无效；Q\_S等于1时传入0和1之外的其他值会执行异常。

- **softmax_lse_flag** (`bool`)：可选参数。表示是否输出softmax\_lse，支持S轴外切（增加输出）。True表示输出softmax\_lse，False表示不输出；默认值为False。
- **key_antiquant_mode** (`int`)：可选参数。表示`key`的伪量化方式。默认值为0，取值除了`key_antiquant_mode`为0并且`value_antiquant_mode`为1的场景外，需要与`value_antiquant_mode`一致。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

    Q\_S大于等于2时仅支持传入值为0、1，Q\_S等于1时支持取值0、1、2、3、4、5。

    | key_antiquant_mode | 模式 |
    |:-------------------|:-----|
    | 0 | perchannel（包含pertensor） |
    | 1 | pertoken |
    | 2 | pertensor 叠加 perhead |
    | 3 | pertoken 叠加 perhead |
    | 4 | pertoken 叠加 page attention 模式管理 scale/offset |
    | 5 | pertoken 叠加 per head 并使用 page attention 模式管理 scale/offset |

- **value_antiquant_mode** (`int`)：可选参数。表示`value`的伪量化方式，模式编号与`key_antiquant_mode`一致。默认值为0，取值除了`key_antiquant_mode`为0并且`value_antiquant_mode`为1的场景外，需要与`key_antiquant_mode`一致。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

    Q\_S大于等于2时仅支持传入值为0、1；Q\_S等于1时支持取值0、1、2、3、4、5。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

- **attention\_out** (`Tensor`)：公式中的输出，数据类型支持`float16`、`bfloat16`、`int8`。数据格式支持$ND$。限制：该返回值的D维度与`value`的D保持一致，其余维度需要与入参`query`的shape保持一致。
- **softmax_lse** (`Tensor`)：Ring Attention算法对query乘key的结果，先取max得到softmax\_max。`query`乘`key`的结果减去softmax\_max，再取exp，最后取sum，得到softmax\_sum，最后对softmax\_sum取log，再加上softmax\_max得到的结果。数据类型支持`float32`，`softmax_lse_flag`为True时，一般情况下，输出shape为\(B, Q\_N, Q\_S, 1\)的Tensor，当input\_layout为TND/NTD\_TND时，输出shape为\(T,Q\_N,1\)的Tensor；`softmax_lse_flag`为False时，则输出shape为\[1\]的值为0的Tensor。数据为inf的代表无效数据。关于Ring Attention的说明见[参考资源](#zh-cn_topic_0000001832267082_section28169228374)。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

### 通用约束

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 确定性计算：默认确定性实现。
- 入参为空的处理：算子内部需要判断参数`query`是否为空，如果是空则直接返回空。参数`query`不为空Tensor，参数`key`、`value`为空Tensor（即S2为0），则`attention_out`按照对应shape大小返回全0。`attention_out`为空Tensor时，返回空。
- 参数`key`、`value`中对应tensor的shape需要完全一致；非连续场景下`key`、`value`的tensorlist中的batch只能为1，个数等于`query`的B，N和D需要相等。由于tensorlist限制，非连续场景下B不能大于256。

### input_layout 约束

- 支持 BSH、BSND、BNSD、BNSD_BSND、BSH_NBSD、BSND_NBSD、BNSD_NBSD、TND、TND_NTD、NTD_TND。
- 排布格式带下划线时，下划线左边为输入 query 的 layout，右边为输出 output 的格式，算子内部进行 layout 转换。
- BNSD_BSND：输入 BNSD → 输出 BSND，仅 Q_S > 1。
- 输出 NBSD 格式（BSH_NBSD、BSND_NBSD、BNSD_NBSD）：仅 Q_S > 1 且 Q_S ≤ 16。
- TND/TND_NTD/NTD_TND 场景：需配合 `actual_seq_lengths` 使用，各 Batch 的 S 累加和为 T。
- 典型 shape 见[参数说明](#zh-cn_topic_0000001832267082_section112637109429)中的 input_layout 典型 shape 说明表。

### 基础维度约束

- **Q_S > 1 时：**
    - B 轴 ≤ 65536。输入类型为 INT8 且 D 轴非 32 字节对齐时，B 轴最大 128；输入类型为 FLOAT16 或 BFLOAT16 且 D 轴非 16 字节对齐时，B 轴最大 128。
    - N 轴 ≤ 256，D 轴 ≤ 512。input_layout 为 BSH 或 BSND 时，要求 N×D < 65535。
    - S ≤ 20971520（20M）。长序列可能超时（aicore error: timeout or trap error），建议 S 切分。
    - 典型超时场景：
        - B=1, Q_N=20, Q_S=2097152, D=256, KV_N=1, KV_S=2097152
        - B=1, Q_N=2, Q_S=20971520, D=256, KV_N=2, KV_S=20971520
        - B=20, Q_N=1, Q_S=2097152, D=256, KV_N=1, KV_S=2097152
        - B=1, Q_N=10, Q_S=2097152, D=512, KV_N=1, KV_S=2097152
    - D 轴对齐：包含 INT8 时需 32 对齐，包含 INT4 时需 64 对齐，全为 FLOAT16/BFLOAT16 时需 16 对齐。
- **Q_S = 1 时：**
    - B 轴 ≤ 65536，N 轴 ≤ 256，S 轴 ≤ 262144，D 轴 ≤ 512。
    - query、key、value 全 INT8 场景暂不支持。

### 全量化场景约束（INT8 输入/输出）

- 输入为 INT8、输出为 INT8：`dequant_scale1`、`quant_scale1`、`dequant_scale2`、`quant_scale2` 需同时存在，`quant_offset2` 可选（不传默认 0）。
    - Atlas A2 训练/推理、Atlas A3 训练/推理：输入为 INT8。
- 输入为 INT8、输出为 FLOAT16：`dequant_scale1`、`quant_scale1`、`dequant_scale2` 需同时存在，若存在 `quant_offset2` 或 `quant_scale2` 则报错。
    - Atlas A2 训练/推理、Atlas A3 训练/推理：输入为 INT8。
- 输入全为 FLOAT16/BFLOAT16、输出为 INT8（后量化）：`quant_scale2` 需存在，`quant_offset2` 可选（不传默认 0）。若存在 `dequant_scale1` 或 `quant_scale1` 或 `dequant_scale2` 则报错。
- `quant_scale2` 和 `quant_offset2` 支持 per-tensor / per-channel，数据类型 FLOAT32 / BFLOAT16。若传入 `quant_offset2`，需保证类型和 shape 与 `quant_scale2` 一致。
- per-channel 时 shape 建议：输出 BSH → (1,1,H) 或 (H,)；输出 BNSD → (1,Q_N,1,D) 或 (Q_N,D)；输出 BSND → (1,1,Q_N,D) 或 (Q_N,D)。
- 输出 INT8 且 `quant_scale2` / `quant_offset2` 为 per-channel 时：不支持左 padding、Ring Attention、D 非 32Byte 对齐。
- 输出 INT8 时：不支持 sparse 为 band 且 pre_tokens / next_tokens 为负数。

### 伪量化场景约束

- 支持 per-channel、per-tensor、per-token 三种模式（根据参数 shape 自动判断，dim=1 为 per-tensor，否则 per-channel）：
    - per-channel：shape 为 (2, KV_N, 1, D)（BNSD）/ (2, KV_N, D)（BSND）/ (2, H)（BSH）。数据类型与 query 相同，`antiquant_mode` 置 0。key/value 为 INT8 时支持。
    - per-tensor：shape 为 (2,)。数据类型与 query 相同，`antiquant_mode` 置 0。key/value 为 INT8 时支持。
    - per-token：shape 为 (2, B, KV_S)。数据类型固定为 FLOAT32，`antiquant_mode` 置 1。key/value 为 INT8 时支持。
- 对称量化：`antiquant_offset` 为 None 时执行对称量化。
- 非对称量化：`antiquant_scale` 和 `antiquant_offset` 需同时存在。
- Q_S > 1 时仅支持 FLOAT16 和 FLOAT32（FLOAT32 仅 PageAttention 场景）。
- Q_S = 1 时仅支持 per-channel。
- 建议使用 KV 伪量化参数分离模式（见下文）。

### KV 伪量化分离场景约束

- 当伪量化参数（`antiquant_scale`/`antiquant_offset`）和 KV 分离量化参数（`key_antiquant_scale` 等）同时传入时，以 KV 分离量化参数为准。
- `key_antiquant_mode` 和 `value_antiquant_mode` 取值需保持一致（`key_antiquant_mode=0` 且 `value_antiquant_mode=1` 的场景除外）。
- `key_antiquant_scale` 和 `value_antiquant_scale` 要么都为空，要么都不为空；`key_antiquant_offset` 和 `value_antiquant_offset` 同理。
- 两者都不为空时，shape 需保持一致（上述除外场景同）。

**管理 scale/offset 的量化模式表（Q_S > 1）：**

| 量化模式 | scale/offset 条件 | key/value 条件 | 支持产品 |
|:---------|:------------------|:---------------|:---------|
| perchannel | shape 支持 (KV_N, 1, D) / (KV_N, D) / (H) / (1, KV_N, 1, D) / (1, KV_N, D) / (1, H)，dtype 与 query 相同 | Atlas A2/A3: 仅 INT8 | Atlas A2 训练/推理、Atlas A3 训练/推理 |
| pertoken | shape 均为 (B, KV_S)，dtype=FLOAT32 | Atlas A2/A3: 仅 INT8 | Atlas A2 训练/推理、Atlas A3 训练/推理 |

> **Q_S > 1 附加约束（Atlas A2/A3）：** `key_antiquant_scale` 与 `value_antiquant_scale` 非空时：要求 query 的 S ≤ 16；query dtype=BFLOAT16，key/value dtype=INT8，输出 dtype=BFLOAT16。不支持 tensorlist、左 padding、page attention、prefix。

**管理 scale/offset 的量化模式表（Q_S = 1）：**

| 量化模式 | scale/offset 条件 | key/value 条件 |
|:---------|:------------------|:---------------|
| perchannel | shape 支持 (1, KV_N, 1, D) / (1, KV_N, D) / (1, H)，dtype 与 query 相同 | INT4(INT32) 或 INT8 |
| pertensor | shape 均为 (1,)，dtype 与 query 相同 | 仅 INT8 |
| pertoken | shape 均为 (1, B, KV_S)，dtype=FLOAT32 | INT4(INT32) 或 INT8 |
| pertensor+perhead | shape 均为 (KV_N,)，dtype 与 query 相同 | 仅 INT8 |
| pertoken+perhead | shape 均为 (B, KV_N, KV_S)，dtype=FLOAT32 | INT4(INT32) 或 INT8 |
| pertoken+page attention | shape 均为 (blocknum, blocksize)，dtype=FLOAT32 | 仅 INT8 |
| pertoken+perhead+page attention | shape 均为 (blocknum, KV_N, blocksize)，dtype=FLOAT32 | 仅 INT8 |
| key perchannel + value pertoken | key: (1, KV_N, 1, D) / (1, KV_N, D) / (1, H)，dtype 同 query；value: (1, B, KV_S)，dtype=FLOAT32 | INT4(INT32) 或 INT8；当 INT8 时 query/output 仅 FLOAT16 |

> 以上模式均支持 Atlas A2 训练/推理、Atlas A3 训练/推理产品。

- INT4(INT32) 伪量化场景不支持后量化。
- Q_S=1 时 `int4` 伪量化场景下，PyTorch 入图仅支持 KV int4 拼接成 int32 输入；KV 的 N、D 或 H 为实际值的八分之一（prefix 同理）；int4 伪量化仅支持 D 64 对齐（int32 支持 D 8 对齐）。

### Page Attention 场景约束

- 开启必要条件：`block_table` 存在且有效，`key`、`value` 按 `block_table` 索引在连续内存中排布。此场景下 `key`、`value` 的 `input_layout` 参数无效。
- `block_table` 中填充 blockid，不会校验合法性，需用户自行保证。
- `block_table` 必须为二维：第一维 = B，第二维 ≥ maxBlockNumPerSeq（maxBlockNumPerSeq = 各 batch 中最大 actual_seq_lengths_kv 对应的 block 数量）。
- 必须传入 `actual_seq_lengths_kv`。
- 不支持 tensorlist 场景，不支持左 padding 场景。

#### Q_S > 1 时

- 支持 key/value dtype 为 FLOAT16/BFLOAT16。
- `block_size`：最小 128，最大 512，需为 128 的倍数。
- kv cache 排布：query layout 为 BNSD/TND 时支持 (blocknum, blocksize, H) 和 (blocknum, KV_N, blocksize, D)；layout 为 BSH/BSND 时仅支持 (blocknum, blocksize, H)。
- KV_N×D > 65535 且排布为 (blocknum, blocksize, H) 时拦截报错 → 开启 GQA 或换 (blocknum, KV_N, blocksize, D)。
- blocknum ≥ 各 batch 按 actual_seq_lengths_kv 和 block_size 计算的 block 数量之和。
- 不支持伪量化场景。
- 不支持 query 为 INT8。
- KV_S ≥ maxBlockNumPerSeq × blockSize（传 atten_mask 或 pse_shift 时）。

#### Q_S = 1 时

- 支持 key/value dtype 为 FLOAT16/BFLOAT16/INT8。
- 不支持 query 为 BFLOAT16/FLOAT16 且 key/value 为 INT4(INT32) 的场景（Atlas A2/A3）。
- `block_size`：需传非 0 值，最大 512。FLOAT16/BFLOAT16 时需 16 对齐，INT8 时需 32 对齐，推荐 128。
- kv cache 排布：同 Q_S > 1 约束。(blocknum, KV_N, blocksize, D) 性能通常优于 (blocknum, blocksize, H)，建议优先选择。
- KV_N×D > 64k 且排布为 (blocknum, blocksize, H) 时拦截报错 → 开启 GQA 或换格式。
- key/value shape 所有维度相乘不能超过 INT32 表示范围。
- 开启 atten_mask 且 sparse_mode 不为 2/3/4 时：mask 最后一维 ≥ block_table 第二维 × block_size。
- 开启 pse_shift：pse_shift 最后一维 ≥ block_table 第二维 × block_size。
- pertoken 伪量化或 pertoken+perhead：相应参数 S ≥ block_table 第二维 × block_size。

### GQA 场景约束

- `num_heads` 整除 `num_key_value_heads`，且 `num_heads` / `num_key_value_heads` ≤ 64。
- BSND、BNSD、BNSD_BSND 场景下，`num_key_value_heads` 需与 key/value 的 N 轴 shape 值一致。

#### GQA 伪量化 + NZ 格式场景

- 支持 perchannel 和 pertoken 模式。
- query dtype = BFLOAT16，key/value dtype = INT8；D = 128；Q_S = 1~16。
- input_layout 仅支持 BSH、BSND、BNSD。
- 仅支持 page attention 场景，blockSize 仅支持 128 或 512。
- key/value 仅支持 NZ 输入：[blockNum, KV_N, D/32, blockSize, 32]。
- perchannel: key/value_antiquant_scale dtype = BFLOAT16；pertoken: dtype = FLOAT32。
- key_antiquant_scale / value_antiquant_scale shape：perchannel 时 BSH→[H], BNSD→[KV_N,1,D], BSND→[KV_N,D]；pertoken 时 [B, KV_S]，S ≥ block_table 第二维 × block_size。
- 仅支持 KV 分离；仅支持高性能模式。
- MTP=0：sparse_mode=0 且不传 mask。0<MTP<16：sparse_mode=3 且传 2048×2048 优化 mask。
- 不支持 key/value_antiquant_offset、query_rope/key_rope、左 padding、tensorlist、pse、prefix、后量化。
- (num_heads, num_key_value_heads) 支持组合：(10,1)、(64,8)、(80,8)、(128,16)。

### MLA / RoPE 场景约束

- `query_rope` 和 `key_rope` 要求同时配置或同时不配置。
- `query_rope` 的数据类型、数据格式与 `query` 一致；`key_rope` 与 `key` 一致。
- 当 `query_rope` 和 `key_rope` 非空时，query 的 D 仅支持 512 或 128。

#### D = 512 时

- sparse：支持 0 / 3 / 4。
- query N = 1/2/4/8/16/32/64/128，query_rope D=64，其余维度与 query 一致。
- key N = 1，D = 512，key_rope D = 64，其余维度与 key 一致。
- key/value/key_rope 支持 ND 或 NZ 格式。NZ 时：FP16/BF16 → [blockNum, KV_N, D/16, blockSize, 16]；INT8 → [blockNum, KV_N, D/32, blockSize, 32]。
- input_layout：BSH、BSND、BNSD、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND、TND_NTD。
- 支持 page attention，block_size 支持 16 的倍数且 ≤ 1024。
- 不支持 softmax_lse、左 padding、tensorlist、pse、prefix、伪量化、全量化、后量化。

#### D = 128 时

- input_layout：BSH、BSND、TND、BNSD、NTD、BSH_BNSD、BSND_BNSD、BNSD_BSND、NTD_TND。
- query_rope D=64，key_rope D=64，其余维度与各自对应 Tensor 一致。
- 不支持左 padding、tensorlist、pse、prefix、伪量化、全量化、后量化。

#### TND / TND_NTD / NTD_TND 场景下 MLA 综合约束

- `actual_seq_lengths` 和 `actual_seq_lengths_kv` 必须传入。入参元素数量 ≤ 4096。
- D = 512 时：
    - sparse: 0/3/4。支持 TND、TND_NTD。
    - 支持 page attention（actual_seq_lengths_kv 长度 = key/value batch，各 batch 实际长度 ≤ KV_S）。
    - query N = 1/2/4/8/16/32/64/128，key/value N = 1。
    - 要求 query_rope 和 key_rope ≠ 空，D = 64。
    - 不支持左 padding、tensorlist、pse、prefix、伪量化、全量化。
- D ≠ 512 时：
    - 无 rope：TND 要求 Q_D=K_D=V_D=128，或 Q_D=K_D=192, V_D=128/192；NTD 不支持 V_D=192；NTD_TND 要求 Q_D=K_D=128/192, V_D=128。
    - 有 rope：要求 Q_D=K_D=V_D=128。
    - 支持 TND、NTD、NTD_TND。
    - page attention: blocksize 16 对齐且 ≤ 1024。
    - 不支持左 padding、tensorlist、pse、prefix、伪量化、全量化、后量化。

### Prefix 场景约束

- `key_shared_prefix` 和 `value_shared_prefix` 要么都为空，要么都不为空。
- 都不为空时：两者与 `key`/`value` 维度相同、dtype 一致。
- `key_shared_prefix` shape 第一维 batch 必须为 1。BNSD/BSND 时 N、D 与 key 一致；BSH 时 H 与 key 一致。`value_shared_prefix` 同理。
- `key_shared_prefix` 和 `value_shared_prefix` 的 S 应相等。
- `actual_shared_prefix_len` 存在时 shape=[1]，值 ≤ 公共前缀的 S。
- 公共前缀的 S + key/value 的 S ≤ 原 key/value 的 S 限制。
- 不支持 page attention、左 padding、tensorlist。
- 不支持 qkv 全 INT8。
- sparse 为 0 或 1 且传入 atten_mask 时：S2 ≥ actual_shared_prefix_len + key 的 S。

### 左 Padding 场景约束

#### Query 左 padding（仅 Q_S > 1）

- query 搬运起点 = Q_S - query_padding_size - actual_seq_lengths，搬运终点 = Q_S - query_padding_size。起点 ≥ 0，终点 ≤ Q_S。
- kv_padding_size < 0 时置 0。
- 需与 `actual_seq_lengths` 同时开启。
- 不支持 page attention。

#### KV 左 padding

- Q_S > 1 时：搬运起点 = KV_S - kv_padding_size - actual_seq_lengths_kv，终点 = KV_S - kv_padding_size。起点 ≥ 0，终点 ≤ KV_S。
- Q_S = 1 时：搬运起点 = Smax - kv_padding_size - actual_seq_lengths，终点 = Smax - kv_padding_size。起点或终点 < 0 时返回全 0。
- kv_padding_size < 0 时置 0。
- 需与 `actual_seq_lengths`（Q_S=1）或 `actual_seq_lengths_kv`（Q_S>1）同时开启。
- 不支持 page attention。
- Q_S=1 时：Atlas A2/A3 不支持 Q 为 BF16/F16 且 KV 为 INT4(INT32) 的场景。

### pse_shift 约束

- 支持 query dtype 为 FLOAT16/BFLOAT16/INT8 时使用。
- query 为 FLOAT16 且 pse_shift 存在时，强制走高精度模式。
- Q_S ≥ query 的 S 长度，KV_S ≥ key 的 S 长度（prefix 场景 KV_S ≥ actual_shared_prefix_len + key 的 S）。

### inner_precise 约束

- Q_S > 1 时 sparse_mode 为 0/1 且传自定义 mask → 建议开启行无效（inner_precise=2/3）。
- Q_S = 1 时仅支持 0 和 1。
- BFLOAT16 和 INT8 不区分高精度/高性能；行无效修正对 FP16/BF16/INT8 均生效。

### softmax_lse 约束

- softmax_lse_flag=True 时 shape 为 [B, N, Q_S, 1]（TND/NTD_TND 时为 [T, N, 1]），数据为 inf 代表无效数据。
- softmax_lse_flag=False 时：softmax_lse 传非空则直接返回该 Tensor；传 None 则返回 shape [1] 全 0 Tensor。

### INT8 输出拦截条件

输出为 INT8 且 `quant_offset2` 非 None，且满足以下条件时，会因某几行不参与计算导致精度误差，该场景会被拦截：

- sparse_mode=0：atten_mask 非 None，且每个 batch actual_seq_lengths - actual_seq_lengths_kv - pre_tokens > 0 或 next_tokens < 0
- sparse_mode=1 或 2：不会触发拦截
- sparse_mode=3：每个 batch actual_seq_lengths_kv - actual_seq_lengths < 0
- sparse_mode=4：pre_tokens < 0 或每个 batch next_tokens + actual_seq_lengths_kv - actual_seq_lengths < 0

> 如果希望该场景不被拦截，请在 FIA 接口外部做后量化操作，不要在 FIA 接口内部开启。

### kvCache 反量化约束

- 合成参数场景仅支持 INT8 反量化到 FLOAT16。
- key/value 的 data range 与 antiquant_scale 的 data range 乘积在 (-1, 1) 内时，高性能模式可保证精度；否则需开高精度。

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

### 示例 1：MHA + 非量化 + BNSD（PromptFlashAttention）

**场景说明**：标准 Multi-Head Attention，全量 prefill 场景，无量化。Q_S > 1 触发 PromptFlashAttention 分支。

**关键参数**：
- `num_heads` = 8，`num_key_value_heads` = 0 → MHA（Q_N = KV_N）
- `input_layout` = "BNSD"
- `scale` = 1/√D
- `pre_tokens` / `next_tokens` = 65535 → 不限制稀疏范围

相关约束：[通用约束](#通用约束)、[基础维度约束](#基础维度约束)。

```python
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
scale = 1 / math.sqrt(128.0)
actseqlen = [164]
actseqlenkv = [1024]

# 调用FIA算子
out, _ = torch_npu.npu_fused_infer_attention_score(q, k, v,
    actual_seq_lengths=actseqlen, actual_seq_lengths_kv=actseqlenkv,
    num_heads=8, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535)

print(out)
# 执行上述代码的输出out类似如下
tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
          [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
          [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
          ...
          [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
          [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
          [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
          device='npu:0', dtype=torch.float16)
```

#### 图模式

```python
import torch
import torch_npu
import math
import torchair as tng

from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)

q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
scale = 1 / math.sqrt(128.0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch_npu.npu_fused_infer_attention_score(q, k, v,
            num_heads=8, input_layout="BNSD", scale=scale,
            pre_tokens=65535, next_tokens=65535)

def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        graph_output = model()
    single_op = torch_npu.npu_fused_infer_attention_score(q, k, v,
        num_heads=8, input_layout="BNSD", scale=scale,
        pre_tokens=65535, next_tokens=65535)
    print("single op output:", single_op[0], single_op[0].shape)
    print("graph output:", graph_output[0], graph_output[0].shape)

if __name__ == "__main__":
    MetaInfershape()
```

---

### 示例 2：GQA + Page Attention + 伪量化（IncreFlashAttention）

**场景说明**：Decode 阶段，Q_S=1（逐 token 推理）。GQA 下 KV_N=2 < Q_N=8，KV Cache 以 Page Attention 方式管理，`key`/`value` 为 INT8 并通过伪量化反量化为 FLOAT16 计算。

**关键参数**：
- `num_heads`=8, `num_key_value_heads`=2 → GQA
- `block_table` + `block_size`=128 → 开启 Page Attention
- `antiquant_scale` → per-channel 伪量化（INT8 KV → FP16）
- `antiquant_mode`=0 → per-channel 模式

相关约束：[Page Attention 场景约束](#page-attention-场景约束)、[GQA 场景约束](#gqa-场景约束)、[伪量化场景约束](#伪量化场景约束)。

```python
import torch
import torch_npu
import math

B, N, S, D = 1, 8, 1, 128          # Q_S=1
KV_N = 2
KV_S = 1024
block_size = 128
block_num = (KV_S + block_size - 1) // block_size

# query: BNSD
q = torch.randn(B, N, S, D, dtype=torch.float16).npu()

# key/value: INT8, 连续存储的 Page Attention KV Cache
k = torch.randint(-128, 127, (block_num, KV_N, block_size, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (block_num, KV_N, block_size, D), dtype=torch.int8).npu()

# block_table: (B, max_blocks_per_seq)
block_table = torch.arange(block_num, dtype=torch.int32).unsqueeze(0).npu()

# 伪量化参数: per-channel
antiquant_scale = torch.randn(2, KV_N, 1, D, dtype=torch.float16).npu()

actseqlenkv = [KV_S]
scale = 1 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    antiquant_scale=antiquant_scale, antiquant_mode=0,
    block_table=block_table, block_size=block_size,
    actual_seq_lengths_kv=actseqlenkv,
    num_heads=N, num_key_value_heads=KV_N,
    input_layout="BNSD", scale=scale
)

print(out.shape)  # torch.Size([1, 8, 1, 128])
```

---

### 示例 3：MLA + RoPE + TND

**场景说明**：MLA 场景，独立传入 RoPE 位置编码信息。TND layout，D=128。每个 Batch 的 seqlen 可能不同，通过 `actual_seq_lengths` 区分。

**关键参数**：
- `query_rope` + `key_rope` → MLA 模式
- `input_layout`="TND" → T=∑S 的变长序列模式
- `actual_seq_lengths` 必须传入，元素值表示各 Batch 的 seqlen 累加和

相关约束：[MLA / RoPE 场景约束](#mla--rope-场景约束)。

```python
import torch
import torch_npu
import math

# 假设 3 个 batch，seqlen 分别为 100, 150, 200
act_seq_lens = [100, 250, 450]  # 累加和: 100, 100+150, 100+150+200
T = 450
N, KV_N, D = 8, 1, 128

# TND layout
q = torch.randn(T, N, D, dtype=torch.float16).npu()
k = torch.randn(T, KV_N, D, dtype=torch.float16).npu()
v = torch.randn(T, KV_N, D, dtype=torch.float16).npu()

# RoPE: D=64
q_rope = torch.randn(T, N, 64, dtype=torch.float16).npu()
k_rope = torch.randn(T, KV_N, 64, dtype=torch.float16).npu()

act_seq_lens_kv = [100, 250, 450]
scale = 1 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    query_rope=q_rope, key_rope=k_rope,
    actual_seq_lengths=act_seq_lens,
    actual_seq_lengths_kv=act_seq_lens_kv,
    num_heads=N, num_key_value_heads=KV_N,
    input_layout="TND", scale=scale, sparse_mode=0
)

print(out.shape)  # torch.Size([450, 8, 128])
```

---

### 示例 4：全量化 INT8 in/out + BNSD

**场景说明**：全量化计算，query/key/value 均为 INT8 输入，attention_out 为 INT8 输出。BMM1 后反量化到 FP16 计算 softmax，BMM2 后再量化回 INT8。

**关键参数**：
- `dequant_scale1` / `quant_scale1` → BMM1 反量化 / BMM2 前量化
- `dequant_scale2` / `quant_scale2` / `quant_offset2` → 输出量化
- 四组量化参数缺一不可

相关约束：[全量化场景约束](#全量化场景约束int8-输入输出)。

```python
import torch
import torch_npu
import math

B, N, Q_S, KV_S, D = 1, 8, 164, 1024, 128

# INT8 输入
q = torch.randint(-128, 127, (B, N, Q_S, D), dtype=torch.int8).npu()
k = torch.randint(-128, 127, (B, N, KV_S, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (B, N, KV_S, D), dtype=torch.int8).npu()

# 量化参数: per-tensor
dequant_scale1 = torch.tensor(0.01, dtype=torch.float32).npu()
quant_scale1 = torch.tensor(0.02, dtype=torch.float32).npu()
dequant_scale2 = torch.tensor(0.01, dtype=torch.float32).npu()
quant_scale2 = torch.tensor(0.03, dtype=torch.float32).npu()

actseqlen = [Q_S]
actseqlenkv = [KV_S]
scale = 1 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    actual_seq_lengths=actseqlen, actual_seq_lengths_kv=actseqlenkv,
    dequant_scale1=dequant_scale1, quant_scale1=quant_scale1,
    dequant_scale2=dequant_scale2, quant_scale2=quant_scale2,
    num_heads=N, input_layout="BNSD", scale=scale,
    pre_tokens=65535, next_tokens=65535
)

print(out.shape, out.dtype)  # torch.Size([1, 8, 164, 128]) torch.int8
```

---

### 示例 5：后量化 FP16 in / INT8 out

**场景说明**：query/key/value 均为 BFLOAT16，仅输出端量化为 INT8。无需 BMM1/BMM2 的 dequant/quant 参数，只需 `quant_scale2`。

**关键参数**：
- `quant_scale2` → 输出量化因子（per-tensor）
- 不应传入 `dequant_scale1` / `quant_scale1` / `dequant_scale2`

相关约束：[全量化场景约束](#全量化场景约束int8-输入输出)。

```python
import torch
import torch_npu
import math

B, N, Q_S, KV_S, D = 1, 4, 128, 512, 64

q = torch.randn(B, N, Q_S, D, dtype=torch.bfloat16).npu()
k = torch.randn(B, N, KV_S, D, dtype=torch.bfloat16).npu()
v = torch.randn(B, N, KV_S, D, dtype=torch.bfloat16).npu()

# 输出量化: per-channel (BSH 场景)
quant_scale2 = torch.randn(1, 1, N * D, dtype=torch.float32).npu()

scale = 1 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    quant_scale2=quant_scale2,
    num_heads=N, input_layout="BNSD", scale=scale,
    pre_tokens=65535, next_tokens=65535
)

print(out.shape, out.dtype)  # torch.Size([1, 4, 128, 64]) torch.int8
```

---

### 示例 6：Prefix + 非量化

**场景说明**：共享 prefix 的 KV 场景。前缀部分 `key_shared_prefix`/`value_shared_prefix` 为所有请求共用的 Prompt KV Cache，后续 `key`/`value` 为各请求独有部分。

**关键参数**：
- `key_shared_prefix` / `value_shared_prefix` → 公共前缀 KV
- `actual_shared_prefix_len` → 前缀有效长度
- prefix S + key/value S 需满足各自 S 维度限制

相关约束：[Prefix 场景约束](#prefix-场景约束)。

```python
import torch
import torch_npu
import math

B, N, Q_S, D = 1, 8, 100, 128
prefix_S = 200
kv_S = 800

q = torch.randn(B, N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, N, kv_S, D, dtype=torch.float16).npu()
v = torch.randn(B, N, kv_S, D, dtype=torch.float16).npu()

# 公共前缀
key_prefix = torch.randn(1, N, prefix_S, D, dtype=torch.float16).npu()
value_prefix = torch.randn(1, N, prefix_S, D, dtype=torch.float16).npu()

actseqlen = [Q_S]
actseqlenkv = [kv_S]
actual_shared_prefix_len = [prefix_S]
scale = 1 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    key_shared_prefix=key_prefix, value_shared_prefix=value_prefix,
    actual_shared_prefix_len=actual_shared_prefix_len,
    actual_seq_lengths=actseqlen, actual_seq_lengths_kv=actseqlenkv,
    num_heads=N, input_layout="BNSD", scale=scale,
    pre_tokens=65535, next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 100, 128])
```

## 参考资源<a name="zh-cn_topic_0000001832267082_section28169228374"></a>

- BMM1（Batch Matrix Multiply 1）

  概念：指第一次矩阵乘法，用于计算`attention score`。

  计算公式：
  $$
  attention\_score = query \cdot key^{\top}
  $$

- BMM2（Batch Matrix Multiply 2）

  概念：指第二次矩阵乘法，用于将`attention`权重应用到`value`上得到最终输出。

  计算公式：
  $$
  attention\_out = atten\_prob \cdot value
  $$
  `atten_prob` 为归一化后的注意力权重：

  $$
  atten\_prob = softmax(\cdot)
  $$

- Ring Attention

  概念：一种分布式注意力算法，将长序列按块（block）切分到多个设备上，通过环形通信逐块计算注意力，从而降低单卡显存占用并支持超长序列。在单卡场景下，Ring Attention退化为标准注意力。

  计算逻辑：当 `softmax_lse_flag=True` 时，算子会额外输出 softmax 的 log-sum-exp（LSE）值，用于在分布式环境中合并多个设备的注意力结果。假设两个设备分别计算了部分序列的注意力输出 \(O_1, O_2\) 和对应的LSE值 \(lse_1, lse_2\)，则全局输出为：

  $$
  O_{global} = \frac{e^{lse_1 - lse_{max}} O_1 + e^{lse_2 - lse_{max}} O_2}{e^{lse_1 - lse_{max}} + e^{lse_2 - lse_{max}}}
  $$
  `lse_max`的定义如下：

  $$
  lse_{max} = max(lse_1, lse_2)。
  $$

  应用场景：该技术常用于超大序列推理或训练。
