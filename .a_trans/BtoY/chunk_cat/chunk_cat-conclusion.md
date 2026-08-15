# aten::chunk_cat A5 vs A2/A3 内存差异 · 分析结论

日期：2026-08-15
算子：`aten::_chunk_cat` / `aten::_chunk_cat.out`（PTA 侧 `torch._chunk_cat`）
现象：A5 上申请显存比 A2/A3 多 4G 以上

## 结论（一句话）

**A2/A3 走 `aclnnChunkCat` 单算子融合路径；A5 上 `npu_support_aclnn=false`，回退到 PyTorch 原生组合实现
`at::native::_chunk_cat`（`new_zeros` 全量 padding 拷贝 + `reshape` + `cat` 小算子链），多出的显存即来自 padding
拷贝与中间张量——与"不支持 chunk cat 就走 cat 小算子拼接"的推测一致，但 A5 的禁用是 PTA 适配层
**显式 SoC 门控**（发布线）造成的，不完全等于 CANN 没导出符号。**

## 证据链

### 1. 两侧 PTA 调用链结构相同（无差异）

- 注册：`aten::_chunk_cat` 在 NPU(PrivateUse1) key 上注册了 kernel。
  - A2/A3 侧（torch_npu 2.13 源码）：`npu_native_functions_by_codegen.yaml` 中 `_chunk_cat: impl_ns: op_api`，
    生成 `RegisterNPU.cpp` 的 `wrapper_NPU___chunk_cat` → `op_plugin::_chunk_cat`。
  - A5 侧（本机安装 torch_npu 2.10.0.post1 实测）：
    `torch._C._dispatch_dump_table('aten::_chunk_cat')` 显示
    `PrivateUse1: registered at torch_npu/csrc/aten/RegisterNPU.cpp:34783 [kernel]`；
    `strings libtorch_npu.so` 含 `wrapper_NPU___chunk_cat`、
    `third_party/op-plugin/op_plugin/ops/opapi/ChunkCatKernelOpApi.cpp`、`aclnnChunkCat`、`call aclnnChunkCat failed`。
  → 两边都进入 op-plugin `op_api::_chunk_cat`（ChunkCatKernelOpApi.cpp），注册层无差异。

### 2. 差异点：op-plugin 运行时可用性判定（关键证据）

`op_plugin/ops/opapi/ChunkCatKernelOpApi.cpp`：

```cpp
static bool npu_support_aclnn =
    check_aclnn_kernel_available("aclnnChunkCat")
    && c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950;   // ← 发布线才有这一半
if (!npu_support_aclnn) {
    return at::native::_chunk_cat(tensors, dim, num_chunks);        // ← 回退组合实现
}
... EXEC_NPU_CMD(aclnnChunkCat, tensors, wrapped_dim, num_chunks, result);
```

- `check_aclnn_kernel_available("aclnnChunkCat")`（op_plugin/utils/op_api_common.cpp:801）：
  动态加载 CANN 算子库并 dlsym `aclnnChunkCat` + `aclnnChunkCatGetWorkspaceSize` 两个符号，任一缺失返回 false
  （`static` 缓存，进程内只判定一次）。
- **SoC 门控 `GetSocVersion() < Ascend950`**：
  - 2026-04-01 `e88c27d7 "fix aclnnChunkCat support"` 加入——当时 CANN 侧 A5(950) 无 aclnnChunkCat 内核；
  - **仅 master 在 2026-07-13 `6ef73e39 "Remove Ascend950 version restriction..."` 移除**；
  - 发布分支（2.10/2.13/7.3.0/26.x 等）仍保留 → **A5 上恒 false → 必回退**。
  - A2/A3（910B 等 SoC < 950）：门控通过 + CANN 8.x `libaclnn_ops_infer.so` 导出符号 → 融合路径。

### 3. 回退路径的内存放大机制（PyTorch 源码验证）

`aten/src/ATen/native/native_functions.yaml`：`_chunk_cat` 为 `CompositeExplicitAutograd`（无 NPU kernel 时走原生组合）。
`aten/src/ATen/native/TensorShape.cpp:3331-3360`：

```cpp
_pad_chunk: 每个输入若 cat 维尺寸不能被 num_chunks 整除 →
            tensor.new_zeros(padded_size);                    // 全量 padded 拷贝（≈输入大小）
            padded_tensor.narrow(dim,0,n).copy_(tensor);      // 独立 copy 算子
            padded_tensor.reshape({..., num_chunks, -1});     // view
_chunk_cat:  return at::cat(_pad_chunk(...), wrapped_dim+1);  // 独立 cat 算子，再分配整块输出
```

峰值显存增量 ≈ Σ(padding 拷贝) ≈ Σ(输入大小)（padding 被触发时）+ cat 中间/输出 + 每个小算子的
workspace。例：4×[1, 57344, 8192] bf16 输入共 3.5GB，padding 触发即多 ~3.5GB+，与 4G+ 现象吻合。
aclnnChunkCat 融合路径只分配一次输出 + 少量 workspace，无 padding 拷贝。

### 4. CANN 9.x 侧事实核查（A5 配套 CANN）

- CANN 9.x 把 aclnn 算子库改名拆分：`libaclnn_ops_infer/train/math/sparse/fft/rand.so`
  → `libopapi.so` + `libopapi_nn/math/cv/transformer/ras.so`（+ `libopapi_legacy.so`）。
  op-plugin 已有兼容：扫描 `$ASCEND_HOME_PATH/lib64` 下 `libopapi_*.so` 逐个 dlsym，再兜底旧名。
- 从本机 `Ascend-cann-910b-ops_9.2.0.beta.1` 包内抽出并 `nm -D` 验证：
  **910B 的 `libopapi_legacy.so` 与 `libopapi_math.so` 均导出 `aclnnChunkCat` /
  `aclnnChunkCatGetWorkspaceSize`（T 符号）**；`libopapi_nn.so` 不导出。
- 本机没有 950(A5) 专属 ops 包，**A5 CANN 是否导出该符号需在 A5 服务器实测**（方法见下）。
- 头文件 `aclnnop/aclnn_chunk_cat.h` 在 CANN 9.2.0-beta.1 中存在（API 已声明）。

## 定性结论

1. A5 内存多 4G+ 的直接原因 = `_chunk_cat` 在 A5 上走了 `at::native::_chunk_cat` 组合回退
   （小算子拼接），而非 aclnnChunkCat 融合算子。
2. 回退的触发条件（二选一，取决于 A5 上实际运行的 PTA/op-plugin 版本）：
   - PTA 用**发布线** op-plugin（2026-07-13 之前的任何 release）：`GetSocVersion() < Ascend950`
     门控在 A5 恒 false，**无条件回退**（与 CANN 符号无关）；
   - PTA 用 **master**（07-13 之后）：门控已移除，回退与否取决于 A5 CANN 的 `libopapi*.so`
     是否导出 `aclnnChunkCat`——若 950 包未导出则符号检查失败回退。
3. "A5 还不支持 aclnnChunkCat" 在 4 月~7 月的发布线上是**代码显式确认的事实**（门控即证据）；
   master 07-13 已放开，是否真支持由 A5 CANN 版本决定，需实测确认。

## 待用户在 A5 服务器验证（具体方法）

1. 确认 PTA 版本与 op-plugin 是否带门控：
   `grep -n "Ascend950" <op-plugin>/op_plugin/ops/opapi/ChunkCatKernelOpApi.cpp`
2. 确认 A5 CANN 导出符号：
   `nm -D $ASCEND_HOME_PATH/lib64/libopapi*.so | grep aclnnChunkCat`
3. 运行时看回退日志（check 失败会打 W 级日志）：
   `dlopen %s failed` / `dlsym %s from %s failed` / `ASCEND_HOME_PATH does not get any libopapi_*.so file`
4. profiler（msprof / torch_npu profiler）对比算子序列：
   A2/A3 应为单个 `aclnnChunkCat`；A5 回退时为 `constant_pad_nd/copy/reshape/cat` 小算子序列。
5. 显存复测建议：让 cat 维能被 num_chunks 整除（消除 padding 拷贝）再对比一次，
   若差异大幅收敛可进一步坐实"padding 拷贝"是 4G+ 的主因。

## 分析过程与方法总结

1. **框架两侧调用链对比**：A2/A3 侧读 release 源码（RegisterNPU.cpp 生成物 + codegen yaml）；
   A5 侧直接对已安装 wheel 做二进制取证 + 运行时 dispatch 探测
   （`_dispatch_has_kernel_for_dispatch_key` / `_dispatch_dump_table` / `strings`）。
2. **算子级运行时判定审计**：读 `ChunkCatKernelOpApi.cpp` 的可用性判定与回退分支，
   用 `git log -S "Ascend950"` 追踪门控的加入/移除提交及所在分支。
3. **回退路径内存机理**：读 PyTorch 原生 `_chunk_cat`/`_pad_chunk` 实现，逐条列出分配点并做量级估算。
4. **CANN 侧符号核验**：嵌套 .run 包流式解包（不解压不安装），`nm -D` 直接验证符号导出表。
5. 全程只读：未安装/卸载任何包，未修改任何现有文件；临时解包产物仅落在 /tmp/cann_ops_inspect。
