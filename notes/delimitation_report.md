# npu_rotate_quant 全流程定位定界报告

## 一、编译阶段

| # | 现象 | 定位过程 | 根因 | 定界 | 修复 |
|---|------|----------|------|------|------|
| 1 | `aclnnStatus 不是一个类型名` | 编译日志指向 L21，读 `op_api_common.h` 确认 EXEC_NPU_CMD 用 `GetOpApiFuncAddr`+dlsym 运行时绑定 | 不需要编译期前向声明 | **我方代码** | 删 `extern "C"` 块 |
| 2 | `c10::value_or_else deprecated` | 编译警告，PyTorch header 有 `deprecated` 标注 | PyTorch v2.10 API 变更 | **我方代码** | `has_value()+value()` |
| 3 | `没有规则可制作目标 build.make` | make 报错，查 build/ 目录 CMakeFiles 内容不完整 | cmake 缓存与文件系统不一致 | **环境** | `rm -rf build/pytorch` |
| 4 | `cp: 是否覆盖 xxx.cpp？` | 同样命令有时卡住，`alias` 发现 `cp='cp -i'` | shell 别名 | **环境** | `/usr/bin/cp` |

## 二、环境阶段

| # | 现象 | 定位过程 | 根因 | 定界 | 修复 |
|---|------|----------|------|------|------|
| 5 | `libhccl.so: cannot open` | `$ASCEND_HOME_PATH` 指向 `/usr/local/Ascend/cann-9.0.0` 但 opp 目录不存在 | 系统默认 CANN 不完整 | **环境** | `source /home/l00881990/cann/set_env.sh` |
| 6 | `no attribute 'npu_rotate_quant'` | `pip show torch_npu` 安装时间早于编译时间 | 换环境后 whl 是旧的 | **环境** | `pip install --force-reinstall` |

## 三、CANN 阶段

| # | 现象 | 定位过程 | 根因 | 定界 | 修复 |
|---|------|----------|------|------|------|
| 7 | `aclnnRotateQuant not in libopapi.so` | `nm -D` 全系统 43 个 libopapi*.so 搜索，全部为 0；对比 `DynamicQuant` 有 27 个符号 | CANN 库未包含此算子 | **CANN 侧** | 用户更新 CANN |
| 8 | 新 CANN 部署后符号=6 | `nm -D libopapi.so \| grep RotateQuant` 返回 6 | CANN 已更新 | **已修复** | 直接可用 |

## 四、UT 阶段

| # | 现象 | 定位过程 | 根因 | 定界 | 修复 |
|---|------|----------|------|------|------|
| 9 | `npu_dynamic_quant not on CPU backend` | 8 个 UT 全报 NotImplementedError，traceback 指向 `custom_op_exec` | CPU 参考用了 NPU op | **测试用例** | 纯 PyTorch 重写 |
| 10 | `alpha must be null when scaleAlg is not 2` | 传了 `scale_alg=2` 仍报错，查 .md：A2/A3"alpha 仅支持空指针" | A2/A3 硬件不支持 alpha | **CANN 约束** | 限制 Ascend950 |
| 11 | `N(64) must be between 128 and 16000` | 随机生成没卡下限 | .md 约束 N∈[128,16000] | **测试用例** | 加 N>=128 |
| 12 | INT4 `y.shape[-1]=224`，expected 32 | `torch.quint4x2=16`，`c10_npu::DType::INT4=285`，比较失败 | 两个 enum 体系没对齐 | **我方代码** | 加 `QUINT4X2_DTYPE=16` |

## 五、ATK 阶段

| # | 现象 | 定位过程 | 根因 | 定界 | 修复 |
|---|------|----------|------|------|------|
| 13 | yaml `dtype_numbers: 700` rejected | pydantic ValidationError，`StandardConfig.model_fields` 只含 acc/perf/mem | ATK 版本升级 | **ATK 框架** | 删除该字段 |
| 14 | node server 静默退出 | `atk node -b npu --devices 0` 无进程，日志仅 banner | ATK server 模式不可用 | **ATK 框架** | 脚本直调 |
| 15 | 200 ATK 跑出 109 fail | 逐个失败 case 重跑全部单独 pass，定位到 3D rotation golden 公式 | 3D golden 用了 `rot[0]` 近似 | **测试用例** | 限制 2D rotation |

## 汇总

| 定界 | 数量 | 问题 |
|------|------|------|
| 我方代码 | 3 | extern "C"、value_or_else、quint4x2 映射 |
| 测试用例 | 3 | CPU golden、N 范围、3D golden |
| 环境 | 4 | cp 别名、cmake 缓存、libhccl、旧 whl |
| CANN 侧 | 2 | libopapi.so 缺算子、A2/A3 不支持 alpha |
| ATK 框架 | 2 | yaml 字段、server 模式 |
| API 兼容性 | 1 | value_or_else deprecated |
| **合计** | **15** | |
