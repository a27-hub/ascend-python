# torch\_npu.npu\_kv\_rmsnorm\_rope\_cache<a name="ZH-CN_TOPIC_0000002343094197"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明<a name="zh-cn_topic_0000002236535552_section1023311522369"></a>

### 基本功能
本算子融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算与RoPE（Rotary Position Embedding）位置编码以及更新KVCache的ScatterUpdate操作。<br>根据输入参数中kv与v的使能情况，按每个张量块的维度索引index，提供两种类型的KvRmsRopeCache计算处理。<br>本算子目前仅支持对应特定场景规定的维度尺寸的Key和Value作为合法输入。
<table>
    <tr>
        <th style="text-align: center;">场景类型</th>
        <th style="text-align: center;">情况概要</th>
        <th style="text-align: center;">量化模式</th>
    </tr>
    <tr>
        <td>场景一</td>
        <td>输入token的Key和Value的合法维度尺寸分别为Dv=512, Dk=64，合并在kv中；v=None。<br>主要计算过程与输出对应关系：<li>RmsNorm(Dv) -> v_embed_out -> ckv_cache<li>Rope(Dk) -> k_embed_out -> k_cache</td>
        <td><li>Norm输出模式不支持量化，其余模式支持无量化/对称量化<li>所有输出模式不支持非对称量化</td>
    </tr>
    <tr>
        <td>场景二</td>
        <td>输入token的Key和Value的合法维度尺寸分别为Dv=128, Dk=192，分离存储：kv存储Key，v存储Value。<br>主要计算过程与输出对应关系：<li>Dv -> v_embed_out -> ckv_cache<li>Rope(RmsNorm(Dk)[-64:]) -> k_embed_out -> k_cache</td>
        <td>所有输出模式支持：无量化、对称量化、非对称量化</td>
    </tr>
</table>

### 计算公式
#### Rope计算
-   **输入张量kv拆分**：拆分为两部分，其中B为批次大小，T为序列长度。

    ![](./figures/zh-cn_formulaimage_0000002239561238.png)

-   **RMS归一化**：对rms\_in，应用RMS归一化。

    ![](./figures/zh-cn_formulaimage_0000002239721038.png)

    -   γ∈R^512是可学习的缩放参数。
    -   Ed\[·\]表示沿最后一个维度（维度d=512）的均值。
    -   ε为小常数（如0.00001），防止除以零。
    -   ⊙表示逐元素相乘。

-   **旋转位置编码（RoPE）**
    1.  重塑与转置：将rope\_in塑并转置以准备旋转

        ![](./figures/zh-cn_formulaimage_0000002239561242.png)

    2.  旋转操作：应用旋转位置编码

        ![](./figures/zh-cn_formulaimage_0000002239721042.png)

        -   cos⁡和sin⁡为预计算的旋转角度参数。
        -   RotateHalf\(k\)将k的后半部分元素移至前半部分并取反，后半部分用前半部分的值。具体来说，对于维度d=64：

        ![](./figures/zh-cn_formulaimage_0000002242091560.png)
#### Cache索引模式
index表示从输入token的索引空间(B, N, S)，映射到输出cache中索引的的计算过程：
$$
index: \mathbb{R}^{B \times S} \rightarrow index\_cache
$$
其中，B表示batch_size，即输入样本批量大小；N表示num_head，即token特征的注意力头数；S表示cache_length，即每个输入序列的长度（标准token数量）。<br>对于相同序列位置（即[B,*,S]序列一致）的token，它不同注意力头下的cache数据在输出中的局部索引完全一致，即index只与[B, S]有关。<br><br>本算子支持的输出索引模式，以及各模式下token在输入到输出cache中的索引对应关系如下：<br>
<table>
    <tr>
        <th style="text-align: center;">Cache模式<br>枚举值</th>
        <th style="text-align: center;">index形状</th>
        <th style="text-align: center;">k_cache形状</th>
        <th style="text-align: center;">ckv_cache形状</th>
        <th style="text-align: center;">模式说明</th>
    </tr>
    <tr>
        <td>Norm</td>
        <td>[B, S]</td>
        <td>[B, N, S, Dk]</td>
        <td>[B, N, S, Dv]</td>
        <td>KV-Cache更新模式，cache内token偏移与输入保持一致。k_cache形状为[batch_size, 1, cache_length, Dk]，ckv_cache形状为[batch_size, 1, cache_length, Dv]。index形状为[batch_size, seq_len]，index里的值表示每个Batch下的偏移。</td>
    </tr>
    <tr>
        <td>PA/PA_BNSD</td>
        <td>[B * S]</td>
        <td>[blk_num, blk_size, N, Dk]</td>
        <td>[blk_num, blk_size, N, Dv]</td>
        <td rowspan="1">PagedAttention模式，index按Page分页数量(blk_num)和分页尺寸(blk_size)表示token偏移。k_cache形状为[block_num, block_size, 1, Dk]，ckv_cache形状为[block_num, block_size, 1, Dv]。index形状为[batch_size*seq_len]，index里的值表示每个token的偏移。</td>
    </tr>
    <tr>
        <td>PA_NZ</td>
        <td>[B * S]</td>
        <td>[blk_num, blk_size, N, Dk]</td>
        <td>[blk_num, blk_size, N, Dv]</td>
        <td rowspan="1">PagedAttention模式，Cache数据格式为FRACTAL_NZ。index按Page分页数量(blk_num)和分页尺寸(blk_size)表示token偏移。k_cache形状为[block_num, block_size, 1, Dk]，ckv_cache形状为[block_num, block_size, 1, Dv]。index形状为[batch_size*seq_len]，index里的值表示每个token的偏移。<br>不同量化模式下数据排布不同：<li>非量化模式下：k_cache数据排布为[block_num, rope_size//16, block_size, 1, 16]，ckv_cache数据排布为[block_num, rms_size//16, block_size, 1, 16]</li><li>量化模式下：k_cache数据排布为[block_num, rope_size//32, block_size, 1, 32]，ckv_cache数据排布为[block_num, rms_size//32, block_size, 1, 32]</li></td>
    </tr>
    <tr>
        <td>PA_BLK_BNSD</td>
        <td>[B * ceil(S/blk_size)]</td>
        <td>[blk_num, blk_size, N, Dk]</td>
        <td>[blk_num, blk_size, N, Dv]</td>
        <td>PagedAttention模式，按固定blk_size尺寸切分，每个样本首token对齐到Page分页起点。<br>index中会包含无token对应的空位置，相应索引为-1。</td>
    </tr>
    <tr>
        <td>PA_BLK_NZ</td>
        <td>[B * ceil(S/blk_size)]</td>
        <td>[blk_num, blk_size, N, Dk]</td>
        <td>[blk_num, blk_size, N, Dv]</td>
        <td>PagedAttention模式,按固定blk_size尺寸切分，每个样本首token对齐到Page分页起点，但数据的内存排布为Fractal_NZ。<br>index中会包含无token对应的空位置，相应索引为-1。<br>不同量化模式下数据排布不同：<li>非量化模式下：k_cache数据排布为[block_num, rope_size//16, block_size, 1, 16]，ckv_cache数据排布为[block_num, rms_size//16, block_size, 1, 16]</li><li>量化模式下：k_cache数据排布为[block_num, rope_size//32, block_size, 1, 32]，ckv_cache数据排布为[block_num, rms_size//32, block_size, 1, 32]</li></td>
    </tr>
</table>

#### 场景一
拆出kv尾轴的左半边作为Dv进行RmsNorm计算，右半边作为Dk用于interleaveRope计算，再分别将计算结果按维度索引index写入ckv_cache和k_cache中。
- (1) RmsNorm:
    $$
    x=kv[...,:Dv]
    $$

    $$
    square\_x=x*x
    $$

    $$
    mean\_square\_x=square\_x.mean(dim=-1,keepdim=True)
    $$

    $$
    rms=torch.sqrt(mean\_square\_x+epsilon)
    $$

    $$
    ckv\_cache[index(B, S)] = (x/rms)*gamma
    $$
- (2) interleaveRope:

    $$
    x=kv[...,Dv:]
    $$

    $$
    x1=x[...,::2]
    $$

    $$
    x2=x[...,1::2]
    $$

    $$
    x\_part1=torch.cat((x1,x2),dim=-1)
    $$

    $$
    x\_part2=torch.cat((-x2,x1),dim=-1)
    $$

    $$
    k\_cache[index(B, S)] = x\_part1*cos+x\_part2*sin
    $$

#### 场景二
将kv尾轴直接作为Dk，进行RmsNorm计算后，取左半边进行interleaveRope计算的结果，与右半边拼接，根据index写出到输出到k_cache中；将v的尾轴作为Dv，根据index输出到ckv_cache中。
- (1) RmsNorm+InterleaveRope
    $$
        x=kv[...,:] \\
        square\_x = x * x
    $$
    $$
    mean\_square\_x=square\_x.mean(dim=-1,keepdim=True)
    $$

    $$
    rms=torch.sqrt(mean\_square\_x+epsilon) \\
    y = (x/rms) * gamma
    $$

    $$
    Ck1 = y[\dots, :Dk\_rope]   \\
    Ck2 = y[\dots, Dk\_rope:]
    $$

    $$
    Ck1\_1 = Ck1[\dots, ::2]   \\
    Ck1\_2 = Ck1[\dots, 1::2]
    $$

    $$
    Ck1\_part1 = torch.cat((Ck1\_1, Ck1\_2), dim=-1)   \\
    Ck1\_part2 = torch.cat((-Ck1\_2, Ck1\_1), dim=-1)
    $$
    $$
    Ck\_rope = Ck1\_part1 * cos + Ck1\_part2 * sin  \\
    k\_cache[index(B, S)] = Concat(Ck\_rope, Ck2)
    $$
- (2) Index Scatter:
    $$
        ckv\_cache[index(B, S)] = v[..., :]
    $$

## 函数原型
```Python
torch_npu.npu_kv_rmsnorm_rope_cache_meta(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None,
                                   c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, v=None, epsilon=1e-5,
                                   cache_mode='Norm', is_output_kv=False):
```

## 参数说明<a name="zh-cn_topic_0000002236535552_section1723416525369"></a>

>**说明：**<br> 
>Tensor中shape使用的变量说明：
>-   batch\_size：batch的大小。
>-   seq\_len：sequence的长度。
>-   hidden\_size：表示MLA输入的向量长度，取值仅支持576。
>-   rms\_size：表示RMSNorm分支的向量长度，取值仅支持512。
>-   rope\_size：表示RoPE分支的向量长度，取值仅支持64。
>-   cache\_length：Norm模式下有效，表示KVCache支持的最大长度。
>-   block\_num：PagedAttention模式下有效，表示Block的个数。
>-   block\_size：PagedAttention模式下有效，表示Block的大小。

-   **kv** (`Tensor`)：必选参数，表示输入的特征张量。数据类型支持`bfloat16`、`float16`，数据格式为$BNSD$，要求为4维张量，形状为\[batch\_size, 1, seq\_len, hidden\_size\]，其中hidden\_size=rms\_size\(Dv\)+rope\_size\(Dk\)。
-   **gamma** (`Tensor`)：必选参数，表示RMS归一化的缩放参数。数据类型支持`bfloat16`、`float16`，数据格式为$ND$，要求为1维张量，形状为\[rms\_size\]。
-   **cos** (`Tensor`)：必选参数，表示RoPE旋转位置编码的余弦分量。数据类型支持`bfloat16`、`float16`，数据格式为$ND$，要求为4维张量，形状为\[batch\_size, 1, seq\_len, rope\_size\]。
-   **sin** (`Tensor`)：必选参数，表示RoPE旋转位置编码的正弦分量。数据类型支持`bfloat16`、`float16`，数据格式为$ND$，要求为4维张量，形状为\[batch\_size, 1, seq\_len, rope\_size\]。
-   **index** (`Tensor`)：必选参数，表示缓存索引张量，用于定位`k_cache`和`ckv_cache`的写入位置。数据类型支持`int64`，数据格式为$ND$。shape取决于`cache_mode`。
-   **k\_cache** (`Tensor`)：必选参数，用于存储量化/非量化的键向量。数据类型支持`bfloat16`、`float16`、`int8`，数据格式为$ND$。shape取决于`cache_mode`。
-   **ckv\_cache** (`Tensor`)：必选参数，用于存储量化/非量化的压缩后的kv向。数据类型支持`bfloat16`、`float16`、`int8`，数据格式为$ND$。shape取决于`cache_mode`。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
-   **k\_rope\_scale** (`Tensor`)：可选参数，默认值None，表示k旋转位置编码的量化缩放因子。数据类型支持`float32`，数据格式为$ND$，要求为1维张量，形状为\[rope\_size\]。量化模式下必填。
-   **c\_kv\_scale** (`Tensor`)：可选参数，默认值None，表示压缩后kv的量化缩放因子。数据类型支持`float32`，数据格式为$ND$，要求为1维张量，形状为\[rms\_size\]。量化模式下必填。
-   **k\_rope\_offset** (`Tensor`)：可选参数，默认值None，表示k旋转位置编码量化偏移量。数据类型支持`float32`，数据格式为$ND$，要求为1维张量，形状为\[rope\_size\]。量化模式下必填。
-   **c\_kv\_offset** (`Tensor`)：可选参数，默认值None，表示压缩后kv的量化偏移量。数据类型支持`float32`，数据格式为$ND$，要求为1维张量，形状为\[rms\_size\]。量化模式下必填。
-   **v**(`Tensor`)：可选参数，在类型二的计算中作为immediate scatter的Dv输入来源。支持[非连续的Tensor](common/非连续的Tensor.md)。shape仅支持4维[B,N,S,Dv]，在类型二的计算中[B, N, S]必须与输入kv保持一致。[数据格式](common/数据格式.md)支持ND，必须与输入kv一致。
-   **epsilon** (`float`)：可选参数，默认值1e-5，表示RMS归一化中的极小值，防止除以零。
-   **cache\_mode** (`str`)：可选参数，默认值'Norm'，表示缓存模式，支持的模式有Norm、PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ。
-   **is\_output\_kv** (`bool`)：可选参数，表示是否输出处理后的`k_embed_out`和`y_out`（未量化的原始值），默认值False不输出，仅`cache_mode`在\(PA/PA\_BNSD/PA\_NZ/PA\_BLK\_BNSD/PA\_BLK\_NZ\)模式下有效。

## 返回值说明<a name="zh-cn_topic_0000002236535552_section3234185215368"></a>
-   **k\_cache** (`Tensor`)：和输入`k_cache`的数据类型、维度、数据格式完全一致（本质in-place更新）。
-   **ckv\_cache** (`Tensor`)：和输入`ckv_cache`的数据类型、维度、数据格式完全一致（本质in-place更新）。
-   **k\_embed\_out** (`Tensor`)：由`is_output_kv`控制，表示需要写入到`k_cache`中的数据，在未经量化和Scatter处理的状态。当`is_output_kv`为true时，需输出。shape为4维[B,N,S,Dk]。数据类型与输入`kv`一致。
-   **y\_out** (`Tensor`)：由`is_output_kv`控制，表示需要写入到`ckv_cache`中的数据，在未经量化和Scatter处理的状态。当`is_output_kv`为true时，需输出。shape为4维[B,N,S,Dv]。数据类型与输入`kv`一致。

## 约束说明<a name="zh-cn_topic_0000002236535552_section1523425283618"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.5.1版本）。
* kv/v格式：
    * 目前仅支持BNSD格式的kv/v作为输入，kv和v的数据类型与[B, N, S]维度必须完全一致。
* 参数说明里shape格式说明：
    * B为输入kv的batch size，S为输入kv的sequence length，大小由用户输入场景决定，无明确限制。 
    * N为输入kv的head number。场景一与DeepSeekV3网络结构强相关，仅支持N=1。场景二支持N=1,2,4,8。
    * D为输入kv的head dim。rms_norm计算所需数据Dv和rope计算所需数据Dk由输入kv的D切分而来。故Dk、Dv大小需满足Dk+Dv=D。同时，Dk需满足rope规则。根据rope规则，Dk为偶数。若cacheModeOptional为NZ场景（cacheModeOptional为PA_NZ、PA_BLK_NZ），Dk、Dv需32B对齐。该规则适用于所有计算类型中。
    * 若cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ），BlockSize需32B对齐。
    * 关于上述32B对齐的情形，对齐值由cache的数据类型决定。以BlockSize为例，若cache的数据类型为int8，则需BlockSize%32=0；若cache的数据类型为float16，则需BlockSize%16=0；若kCacheRef与ckvCacheRef参数的dtype不一致，BlockSize需同时满足BlockSize%32=0和BlockSize%16=0。
    * Bcache为输入cache的batch size，Scache为输入cache的sequence length，大小由用户输入场景决定，无明确限制。 
    * BlockNum为写入cache的内存块数，大小由用户输入场景决定，无明确限制。 
* index相关约束：
    * 当cacheModeOptional为Norm时，shape为2维[B,S]，要求index的value值范围为[-1,Scache)。不同的B下，value数值可以重复。
    * 当cacheModeOptional为PA_BNSD、PA_NZ时，shape为1维[B * S]，要求index的value值范围为[-1,BlockNum * BlockSize)。value数值不能重复。
    * 当cacheModeOptional为PA_BLK_BSND、PA_BLK_NZ时，shape为1维[B * ceil_div(S,BlockSize)]，要求index的value的数值范围为[-1,BlockNum * BlockSize)。value/BlockSize的值不能重复。
* cos/sin项相关约束：
    * cos/sin最后一维大小总是和当前场景需要进行Rope计算的输入维度一致。
    * 在场景一中，仅Dk部分全部参与Rope运算，因而cos/sin的大小为[B,N,S,Dk]，Dk=64。
    * 在场景二中，Dk仅前64维参与Rope运算，因而cos/sin的大小为[B,N,S,64]。
* v相关约束:
    * 该参数仅在场景二中作为必须入参，在场景一中会作为无效参数被忽略。
    * 如果需要调用场景二，则必须传入尺寸为[B,N,S,128]的Tensor作为v。
-   量化模式：当`k_rope_scale`和`c_kv_scale`非空时，`k_cache`和`ckv_cache`的dtype为`int8`，缓存形状的最后一个维度需要为32（Cache数据格式为FRACTAL\_NZ模式），`k_rope_scale`和`c_kv_scale`必须同时非空，`k_rope_offset`和`c_kv_offset`必须同时为None为非空。
-   非量化模式：当`k_rope_scale`和`c_kv_scale`为空时，`k_cache`和`ckv_cache`的dtype为`bfloat16`或`float16`。
-   索引映射：所有`cache_mode`缓存模式下，index的值不可以重复，如果传入的index值存在重复，算子的行为是未定义的且不可预知的。
    -   Norm：index的值表示每个Batch下的偏移。
    -   PA/PA\_BNSD/PA\_NZ：index的值表示全局的偏移。
    -   PA\_BLK\_BNSD/PA\_BLK\_NZ：index的值表示每个页的全局偏移；这个场景假设cache更新是连续的，不支持非连续更新的cache。

-   Shape关联规则：不同的`cache_mode`缓存模式有不同的Shape规则。
    -   Norm：k\_cache形状为\[batch\_size, 1, cache\_length, rope\_size\]，ckv\_cache形状为\[batch\_size, 1, cache\_length, rms\_size\]，index形状为\[batch\_size, seq\_len\], cache\_length\>=seq\_len。
    -   非Norm模式\(PagedAttention相关模式\)：要求block\_num\>=Ceil\(seq\_len/block\_size\)\*batch\_size。

## 调用示例<a name="zh-cn_topic_0000002236535552_section3235105212365"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    
    batch_size=8
    seq_len=1
    page_num=8
    page_size=128
    input_dtype = torch.float16
    
    kv = torch.randn(batch_size, 1, seq_len, 576, dtype = input_dtype).npu()
    gamma = torch.randn(512, dtype = input_dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = input_dtype).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    is_output_kv = True
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_cache, v_cache, k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index,
                                                                                 k_cache, ckv_cache,
                                                                                 k_rope_scale=k_rope_scale,
                                                                                 c_kv_scale=c_kv_scale,
                                                                                 k_rope_offset=k_rope_offset,
                                                                                 c_kv_offset=c_kv_offset,
                                                                                 epsilon=epsilon,
                                                                                 cache_mode=cache_mode,
                                                                                 is_output_kv=is_output_kv)
            return k_cache, v_cache, k_rope, c_kv
    
    model = Model().npu()
    _, _, k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)
    
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    batch_size=8
    seq_len=1
    page_num=8
    page_size=128
    input_dtype = torch.float16
    
    kv = torch.randn(batch_size, 1, seq_len, 576, dtype = input_dtype).npu()
    gamma = torch.randn(512, dtype = input_dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = input_dtype).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    is_output_kv = True
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_cache, v_cache, k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index,
                                                                                 k_cache, ckv_cache,
                                                                                 k_rope_scale=k_rope_scale,
                                                                                 c_kv_scale=c_kv_scale,
                                                                                 k_rope_offset=k_rope_offset,
                                                                                 c_kv_offset=c_kv_offset,
                                                                                 epsilon=epsilon,
                                                                                 cache_mode=cache_mode,
                                                                                 is_output_kv=is_output_kv)
            return k_cache, v_cache, k_rope, c_kv
    
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    _, _, k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)
    ```

