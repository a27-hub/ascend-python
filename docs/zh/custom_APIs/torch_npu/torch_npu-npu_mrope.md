# torch_npu.npu_mrope

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |

> [!NOTE]
> 本接口在PyTorch侧通过[aclnnRopeWithSinCosCacheV2](https://gitcode.com/cann/ops-transformer/blob/master/posembedding/rope_with_sin_cos_cache/docs/aclnnRopeWithSinCosCacheV2.md)等底层算子实现；产品支持范围与上述CANN算子一致。若运行环境不支持`aclnnRopeWithSinCosCacheV2`，使用`cache_mode='interleave'`时会报错，需升级CANN（参考）或改用`cache_mode='default'`（将回退至`aclnnRopeWithSinCosCache`）。

## 功能说明

- API功能：推理场景下将sin、cos以预计算cache形式传入，对`query`、`key`做旋转位置编码（RoPE）。支持普通**RoPE**（一维`positions`）与多模态**MRoPE**（二维`positions`，按`mrope_section`整合多段位置编码），`cache_mode='default'`为分段式拼接，`cache_mode='interleave'`为交错式拼接。

- 计算公式（与[aclnnRopeWithSinCosCacheV2](https://gitcode.com/cann/ops-transformer/blob/master/posembedding/rope_with_sin_cos_cache/docs/aclnnRopeWithSinCosCacheV2.md)一致；`rotaryDim`为`cos_sin_cache`最后一维大小）：

    **1、从cache取cos、sin**

    - RoPE：`positions`形状为`(numTokens)`。
      $$cosSin[i] = cosSinCache[positions[i]]$$
      $$cos, sin = cosSin.chunk(2, dim=-1)$$

    - MRoPE：`positions`形状为$(m, numTokens)$，$m$为`mropeSection`元素个数（3或4）。先按行从`cosSinCache`取行并`chunk`得到各行的`cos`、`sin`，再按`cacheMode`与`mropeSection`合并为与token对齐的`cos`、`sin`（分段或交错拼接规则见[CANN文档](https://gitcode.com/cann/ops-transformer/blob/master/posembedding/rope_with_sin_cos_cache/docs/aclnnRopeWithSinCosCacheV2.md)；当$m=3$且`cacheMode=0`时，对`cos`有
      $cos = cat(cos_{0,:,0:s_0},\, cos_{1,:,s_0:s_0+s_1},\, cos_{2,:,s_0+s_1:s_0+s_1+s_2})$，其中$s=mropeSection$，$sin$同理）。

    **2、拆分旋转维与直通维**

    $$queryRot = query[..., :rotaryDim]$$
    $$queryPass = query[..., rotaryDim:]$$
    $$keyRot = key[..., :rotaryDim]$$
    $$keyPass = key[..., rotaryDim:]$$

    **3、旋转（`query`与`key`共用同一套`cos`、`sin`广播）**

    - rotate_half（GPT-NeoX，`rotary_mode='half'`）：
      $$
      x_1^q, x_2^q = chunk(queryRot,\, 2,\, dim=-1),\quad x_1^k, x_2^k = chunk(keyRot,\, 2,\, dim=-1)
      $$
      $$
      o_1^q = x_1^q \odot cos - x_2^q \odot sin,\quad o_2^q = x_2^q \odot cos + x_1^q \odot sin
      $$
      $$
      o_1^k = x_1^k \odot cos - x_2^k \odot sin,\quad o_2^k = x_2^k \odot cos + x_1^k \odot sin
      $$
      $$
      queryRot \leftarrow cat(o_1^q, o_2^q,\, dim=-1),\quad keyRot \leftarrow cat(o_1^k, o_2^k,\, dim=-1)
      $$

    - rotate_interleaved（GPT-J，`rotary_mode='interleave'`）：
      $$
      x_1^q = queryRot[..., ::2],\; x_2^q = queryRot[..., 1::2],\quad x_1^k = keyRot[..., ::2],\; x_2^k = keyRot[..., 1::2]
      $$
      $$
      o_1^q = x_1^q \odot cos - x_2^q \odot sin,\quad o_2^q = x_2^q \odot cos + x_1^q \odot sin
      $$
      $$
      o_1^k = x_1^k \odot cos - x_2^k \odot sin,\quad o_2^k = x_2^k \odot cos + x_1^k \odot sin
      $$
      $$
      queryRot \leftarrow stack(o_1^q, o_2^q,\, dim=-1).flatten(-2),\quad keyRot \leftarrow stack(o_1^k, o_2^k,\, dim=-1).flatten(-2)
      $$

    **4、输出**

    $$query = cat(queryRot, queryPass,\, dim=-1)$$
    $$key = cat(keyRot, keyPass,\, dim=-1)$$

- 等价计算逻辑（CPU侧参考实现，与算子数学含义一致；MRoPE仅给出三段、`cache_mode='default'`的合并分支，四段及`interleave`见CANN文档）：

    ```python
    import torch

    def _rotate(query_rot, cos, sin, neox_style):
        # query_rot: (num_tokens, num_heads, rotary_dim)
        # cos, sin: (num_tokens, 1, rotary_dim // 2)
        if neox_style:
            x1, x2 = torch.chunk(query_rot, 2, dim=-1)
        else:
            x1, x2 = query_rot[..., ::2], query_rot[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if neox_style:
            return torch.cat((o1, o2), dim=-1)
        return torch.stack((o1, o2), dim=-1).flatten(-2)

    def npu_mrope_equivalent(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        mrope_section=None,
        neox_style=True,
        cache_mode="default",
    ):
        """positions: 1D(num_tokens,)为RoPE；2D(m, num_tokens)为MRoPE。"""
        rotary_dim = cos_sin_cache.shape[-1]
        num_tokens = positions.shape[-1]
        if positions.ndim == 1:
            cos_sin = cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)
        else:
            flat = positions.flatten()
            cos_sin = cos_sin_cache.index_select(0, flat).reshape(-1, rotary_dim)
            cos, sin = cos_sin.chunk(2, dim=-1)
            cos = cos.reshape(positions.shape[0], -1, rotary_dim // 2)
            sin = sin.reshape(positions.shape[0], -1, rotary_dim // 2)
            if cache_mode == "default" and positions.shape[0] == 3:
                s0, s1, s2 = mrope_section[0], mrope_section[1], mrope_section[2]
                cos = torch.cat(
                    (
                        cos[0, :, :s0],
                        cos[1, :, s0 : s0 + s1],
                        cos[2, :, s0 + s1 : s0 + s1 + s2],
                    ),
                    dim=-1,
                )
                sin = torch.cat(
                    (
                        sin[0, :, :s0],
                        sin[1, :, s0 : s0 + s1],
                        sin[2, :, s0 + s1 : s0 + s1 + s2],
                    ),
                    dim=-1,
                )
            else:
                raise NotImplementedError("见CANN文档中MRoPE四段或cache_mode=interleave分支")

        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        q_shape, k_shape = query.shape, key.shape
        query = query.view(num_tokens, -1, head_size)
        key = key.view(num_tokens, -1, head_size)
        q_rot = query[..., :rotary_dim]
        q_pass = query[..., rotary_dim:]
        k_rot = key[..., :rotary_dim]
        k_pass = key[..., rotary_dim:]
        q_rot = _rotate(q_rot, cos, sin, neox_style)
        k_rot = _rotate(k_rot, cos, sin, neox_style)
        query = torch.cat([q_rot, q_pass], dim=-1).reshape(q_shape)
        key = torch.cat([k_rot, k_pass], dim=-1).reshape(k_shape)
        return query, key
    ```

## 函数原型

```python
torch_npu.npu_mrope(positions, query, key, cos_sin_cache, head_size, *, mrope_section=None, rotary_mode='half', cache_mode='default') -> (Tensor, Tensor)
```

## 参数说明

- **positions**（`Tensor`）：位置索引，用于从cache中选取位置编码。  
  - **RoPE**：1D，shape`(numTokens)`，类型`int32`或`int64`，格式`ND`，可非连续。  
  - **MRoPE**：2D，shape`(3, numTokens)`或`(4, numTokens)`，要求与`mrope_section`分段数一致。  
  - 通过`positions`从`cos_sin_cache`中选取位置编码时，每个取值作为cache行索引，应均小于第0维长度`max_seq_len`。
- **query**（`Tensor`）：待施加RoPE的Query，2D，shape`(num_tokens, num_q_heads * head_size)`，类型`float16`、`bfloat16`或`float32`，格式`ND`，可非连续。
- **key**（`Tensor`）：待施加RoPE的Key，2D，shape`(num_tokens, num_k_heads * head_size)`，类型与`query`一致，格式`ND`，可非连续。
- **cos_sin_cache**（`Tensor`）：预计算的位置编码cache，2D，shape`(max_seq_len, rotary_dim)`。最后一维经`chunk(2, dim=-1)`拆成cos、sin两半。`query`、`key`、`cos_sin_cache`的浮点类型须一致。
- **head_size**（`int`）：单头维度大小，即每个注意力头的特征维长度。
- **mrope_section**（`int[]`，可选）：MRoPE各段在half旋转维度上的长度配置；不传或`[0, 0, 0]`表示**RoPE模式**。使能MRoPE时需与`positions`的行数及`rotary_dim`约束一致（见约束说明）。
- **rotary_mode**（`str`，可选）：`half`表示NeoX风格，`interleave`表示GPT-J风格。默认`half`。
- **cache_mode**（`str`，可选）：`default`对应分段式cos/sin拼接（`cacheMode=0`）；`interleave`对应交错式拼接（`cacheMode=1`）。默认`default`。

## 返回值说明

- **query_out**（`Tensor`）：对`query`施加RoPE/MRoPE后的Query，2D，shape`(num_tokens, num_q_heads * head_size)`，类型`float16`、`bfloat16`或`float32`，格式`ND`，可非连续，与输入`query`的shape与dtype一致。
- **key_out**（`Tensor`）：对`key`施加RoPE/MRoPE后的Key，2D，shape`(num_tokens, num_k_heads * head_size)`，类型与输入`key`一致，格式`ND`，可非连续，与输入`key`的shape与dtype一致。

## 约束说明

- **维度**：`query`、`key`、`cos_sin_cache`均为2维；`rotary_dim`为`cos_sin_cache`最后一维大小，且满足`rotary_dim <= head_size`。
- **对齐与倍数**：
  - `head_size`：`float16`/`bfloat16`时为**32**的倍数；`float32`时为**16**的倍数。
  - `rotary_dim`：`float16`/`bfloat16`时为**32**的倍数；`float32`时为**16**的倍数。
- **MRoPE**：`mrope_section`各元素之和应等于**`rotary_dim / 2`**（与half维度的cos/sin分段一致）。
- **MRoPE支持的`mrope_section`取值**：当前支持`[16, 24, 24]`、`[24, 20, 20]`、`[8, 12, 12]`、`[16, 16, 16, 16]`。
- **cache_mode**：MRoPE下仅支持`default`与`interleave`。当`mrope_section`为**`[16, 16, 16, 16]`**时，**仅支持`cache_mode='default'`**（即不支持交错拼接）。

## 调用示例

### MRoPE示例（三段`mrope_section`）

`positions`为二维`(3, num_tokens)`，与`[16, 24, 24]`等三段配置对应；`mrope_section`三数之和须等于`rotary_dim / 2`。

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_mrope = torch.arange(num_tokens, dtype=torch.int64).repeat(3, 1).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

mrope_section_3 = [16, 24, 24]
query_out, key_out = torch_npu.npu_mrope(
    positions_mrope,
    query,
    key,
    cos_sin_cache,
    head_size,
    mrope_section=mrope_section_3,
    rotary_mode="half",
    cache_mode="default",
)
```

### MRoPE示例（四段`mrope_section`）

`positions`为`(4, num_tokens)`，与`[16, 16, 16, 16]`等四段配置对应；该配置下**`cache_mode`仅支持`'default'`**。

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_mrope4 = torch.arange(num_tokens, dtype=torch.int64).repeat(4, 1).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

mrope_section_4 = [16, 16, 16, 16]
query_out, key_out = torch_npu.npu_mrope(
    positions_mrope4,
    query,
    key,
    cos_sin_cache,
    head_size,
    mrope_section=mrope_section_4,
    rotary_mode="half",
    cache_mode="default",
)
```

### RoPE示例

使用一维`positions`，shape为`(num_tokens,)`；`mrope_section`置为`[0, 0, 0]`表示不使能MRoPE（与当前`torch_npu`默认行为一致）。

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_1d = torch.arange(num_tokens, dtype=torch.int64).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

query_out, key_out = torch_npu.npu_mrope(
    positions_1d,
    query,
    key,
    cos_sin_cache,
    head_size,
    mrope_section=[0, 0, 0],
    rotary_mode="half",
    cache_mode="default",
)
```
