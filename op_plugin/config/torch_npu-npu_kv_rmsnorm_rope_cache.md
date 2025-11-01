# torch_npu-npu_top_k_top_p_sample.md

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

### 基本功能
<br>本算子根据输入参数中kv与v的使能情况，按每个张量块的维度索引index，提供两种类型的KvRmsRopeCache计算处理。<br>本算子目前仅支持对应特定场景规定的维度尺寸的Key和Value作为合法输入。
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
#### Cache索引模式
index表示从输入token的索引空间(B, N, S)，映射到输出cache中索引的的计算过程：
$$
index: \mathbb{R}^{B \times S} \rightarrow index\_cache
$$
其中，B表示batch_size，即输入样本批量大小；N表示num_head，即token特征的注意力头数；S表示cache_length，即每个输入序列的长度（标准token数量）。<br>对于相同序列位置（即[B,*,S]序列一致）的token，它不同注意力头下的cache数据在输出中的局部索引完全一致，即index只与[B, S]有关。<br><br>本算子支持的输出索引模式，以及各模式下token在输入到输出cache中的索引对应关系如下：<br>
<table>
    <tr>
        <th style="text-align: center;">Cache模式</th>
        <th style="text-align: center;">index形状</th>
        <th style="text-align: center;">k_cache形状</th>
        <th style="text-align: center;">ckv_cache形状</th>
        <th style="text-align: center;">说明</th>
    </tr>
    <tr>
        <td>Norm</td>
        <td>[B, S]</td>
        <td>[B, N, S, Dk]</td>
        <td>[B, N, S, Dv]</td>
        <td>kv-cache模式，cache内token偏移与输入保持一致。</td>
    </tr>
    <tr>
        <td>PA/PA_BNSD</td>
        <td>[B * S]</td>
        <td>[blk_num, blk_size, N, Dk]</td>
        <td>[blk_num, blk_size, N, Dv]</td>
        <td>PagedAttention模式。<br>index按Page分页数量(blk_num)和分页尺寸(blk_size)表示token偏移。</td>
    </tr>
    <tr>
        <td>PA_NZ</td>
        <td>[B * S]</td>
        <td>[blk_num, blk_size, N, Dk]</td>
        <td>[blk_num, blk_size, N, Dv]</td>
        <td>PagedAttention模式，但数据的内存排布为Fractal_NZ。<br>index按Page分页数量(blk_num)和分页尺寸(blk_size)表示token偏移。</td>
    </tr>
    <tr>
        <td>PA_BLK_BNSD</td>
        <td>[B * ceil(S/blk_size)]</td>
        <td>[blk_num, blk_size, N, Dk]</td>
        <td>[blk_num, blk_size, N, Dv]</td>
        <td>按固定blk_size尺寸切分，每个样本首token对齐到Page分页起点的PagedAttention模式。<br>index中会包含无token对应的空位置，相应索引为-1。</td>
    </tr>
    <tr>
        <td>PA_BLK_NZ</td>
        <td>[B * ceil(S/blk_size)]</td>
        <td>[blk_num, blk_size, N, Dk]</td>
        <td>[blk_num, blk_size, N, Dv]</td>
        <td>按固定blk_size尺寸切分，每个样本首token对齐到Page分页起点的PagedAttention模式，但数据的内存排布为Fractal_NZ。<br>index中会包含无token对应的空位置，相应索引为-1。</td>
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
## 参数说明
* kv(Tensor，计算输入)：必选参数，公式中用于切分出rms_norm计算所需数据Dv和rope计算所需数据Dk的输入数据。支持[非连续的Tensor](common/非连续的Tensor.md)。shape仅支持4维[B,N,S,D]。[数据格式](common/数据格式.md)支持ND，数据类型支持FLOAT6、BFLOAT16。
* gamma(Tensor，计算输入)：必选参数，公式中用于rms_norm计算的输入数据。支持[非连续的Tensor](common/非连续的Tensor.md)。shape为1维[Dv,]。[数据格式](common/数据格式.md)支持ND。数据类型与输入kv一致。
* cos(Tensor，计算输入)：必选参数，公式中用于rope计算的输入数据，对输入张量Dk进行余弦变换，支持[非连续的Tensor](common/非连续的Tensor.md)。shape为4维[B,N,S,D_rope]或[B,N,1,D_rope]，[数据格式](common/数据格式.md)支持ND。数据类型与输入kv一致。
* sin(Tensor，计算输入)：必选参数，公式中用于rope计算的输入数据，对输入张量Dk进行正弦变换。支持[非连续的Tensor](common/非连续的Tensor.md)。shape为4维[B,N,S,D_rope]或[B,N,1,D_rope]，与cos的shape保持一致。[数据格式](common/数据格式.md)支持ND。数据类型与输入kv一致。
* index(Tensor，计算输入)：必选参数，用于指定写入cache的具体索引位置，当index的value数值为-1时，代表跳过更新。支持[非连续的Tensor](common/非连续的Tensor.md)。当cache_mode为Norm时，shape为2维[B,S]；当cache_mode为PA_BNSD、PA_NZ时，shape为1维[B * S]；当cache_mode为PA_BLK_BSND、PA_BLK_NZ时，shape为1维[B\*ceil_div(S,BlockSize)]。[数据格式](common/数据格式.md)支持ND。数据类型支持INT64。 
* k_cache(Tensor，计算输入/输出)：必选参数，提前申请的cache，输入输出同地址复用。支持[非连续的Tensor](common/非连续的Tensor.md)。当cache_mode为PA场景（cache_mode为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）时，shape为4维[BlockNum,BlockSize,N,Dk]；当cache_mode为Norm场景时，shape为4维[Bcache,N,Scache,Dk]。[数据格式](common/数据格式.md)支持ND。数据类型与输入kv一致或者INT8。
* ckv_cache(Tensor，计算输入/输出)：必选参数，提前申请的cache，输入输出同地址复用。支持[非连续的Tensor](common/非连续的Tensor.md)。当cache_mode为PA场景（cache_mode为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）时，shape为4维[BlockNum,BlockSize,N,Dv]；当cache_mode为Norm场景时，shape为4维[Bcache,N,Scache,Dv]。[数据格式](common/数据格式.md)支持ND。数据类型与输入kv一致或者INT8。
* k_rope_scale(Tensor，计算输入)：可选参数，当k_cache数据类型为INT8时需要此输入参数。支持[非连续的Tensor](common/非连续的Tensor.md)。shape为2维[N,Dk]；或者shape为1维[Dk,]；或者shape为1维[1,]。[数据格式](common/数据格式.md)支持ND。数据类型支持FLOAT32。
* c_kv_scale(Tensor，计算输入)：可选参数，当ckv_cache数据类型为INT8时需要此输入参数。支持[非连续的Tensor](common/非连续的Tensor.md)。shape为2维[N,Dv]；或者shape为1维[Dv,]；或者shape为1维[1,]。[数据格式](common/数据格式.md)支持ND。数据类型支持FLOAT32。
* k_rope_offset(Tensor，计算输入)：可选参数。当k_cache数据类型为INT8且对应的k_rope_scale输入存在并量化场景为非对称量化时，需要此参数输入。支持[非连续的Tensor](common/非连续的Tensor.md)。shape为2维[N,Dk]；或者shape为1维[Dk,]；或者shape为1维[1,]。[数据格式](common/数据格式.md)支持ND。数据类型支持FLOAT32。
* c_kv_offset(Tensor，计算输入)：可选参数，当ckv_cache数据类型为INT8且对应的c_kv_scale输入存在并量化场景为非对称量化时，需要此参数输入。支持[非连续的Tensor](common/非连续的Tensor.md)。shape为2维[N,Dv]；或者shape为1维[Dv,]；或者shape为1维[1,]。[数据格式](common/数据格式.md)支持ND。数据类型支持FLOAT32。
* v(Tensor，计算输入)：可选参数，在类型二的计算中作为immediate scatter的Dv输入来源。支持[非连续的Tensor](common/非连续的Tensor.md)。shape仅支持4维[B,N,S,Dv]，在类型二的计算中[B, N, S]必须与输入kv保持一致。[数据格式](common/数据格式.md)支持ND，必须与输入kv一致。
* epsilon(double，输入)：必选参数，rms_norm计算防止除0。float类型浮点数。建议设为1e-5。
* cache_mode(char\*，输入)：必选参数，cache格式的选择标记。char\*类型。类型有Norm、PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ，建议设为Norm。
* is_output_kv(bool，输入)：必选参数，k_cache_prec和ckv_cache_prec输出控制标记。bool类型。当is_output_kv为true时，表示需输出k_cache_prec和ckv_cache_prec。建议设为false。

## 返回值说明
* k_cache(Tensor，计算输入/输出)：原地更新参数，输入输出同地址复用。具体情况与入参一致。
* ckv_cache(Tensor，计算输入/输出)：原地更新参数，输入输出同地址复用。具体情况与入参一致。
* k_cache_prec(Tensor，计算输出)：由is_output_kv控制，表示需要写入到k_cache中的数据，在未经量化和Scatter处理的状态。当is_output_kv为true时，需输出。shape为4维[B,N,S,Dk\_rope]。数据类型与输入kv一致。
* ckv_cache_prec(Tensor，计算输出)：由is_output_kv控制，表示需要写入到ckv_cache中的数据，在未经量化和Scatter处理的状态。当is_output_kv为true时，需输出。shape为4维[B,N,S,Dk\_rope]。数据类型与输入kv一致。
## 约束说明
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

## 调用示例
* 请后续同事根据aclnn原设文档补充