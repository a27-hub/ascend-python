# GPU 双标杆 · nodes.yaml 与远端配置

> **配套**：[`.md`](.md)（跳板 / 防火墙 / `atk server`）  
> **纪律**：[hard-constraints · 统一口径](../knowledge/hard-constraints.md#统一口径--外部资源获取方式) — host/port/path **由用户粘贴** yaml 或操作信息原文；URL 仅存档。

---

## 1. 必须替换的占位项

模板 yaml 里下列字段 **几乎总是占位**，直接跑必失败：

| 字段 | 说明 |
|------|------|
| `backend: gpu` 的 **host** | 实际可达 GPU 机或跳板暴露地址（非文档示例 IP） |
| **port** | 与 GPU 侧 `atk server -p` **及** SSH 转发口一致 |
| **output_path** | GPU 端 ATK 写产物的目录；**必须与 GPU `atk server` 启动时工作目录/配置一致** |
| **devices** | 与用户实际卡号一致 |
| NPU 侧 **case_json / yaml 路径** | 本地 `ATK_CIDA` 或容器内真实路径（见 op-atk-adapt 路径纪律） |

---

## 2. output_path 不一致 → NPU 拉 GPU 结果 404

双标杆模式下 NPU 端 ATK 会 **拉取 GPU 侧 output**。若 yaml 里 GPU `output_path` 与 GPU 机器上 `atk server` 实际写入目录不一致，表现为 **404 / 下载失败 / 精度列空**。

**检查清单**（用户确认后写入 `运行记录.md`）：

- [ ] GPU 上 `atk server` 进程的工作目录 = yaml 中 `output_path` 前缀  
- [ ] NPU 端 nodes 里 **同一 output_path 字符串**（含尾斜杠习惯一致）  
- [ ] 跳板场景：NPU 填的是 **Windows 141.* 地址 + 转发端口**，不是 GPU 内网裸 IP（见 ）

**正确配置示例（占位符）**：

```yaml
nodes:
  - backend: npu
    task: ["accuracy"]
    devices: [0]
  - backend: gpu
    host: <用户确认的_GPU_或跳板_host>
    port: <与_atk_server_一致>
    output_path: <GPU_端_ATK_实际写盘目录>/
    devices: [0]
    task: ["accuracy"]
```

---

## 3. NPU 端执行前基线

与 [op-atk-adapt · ATK 跑前环境基线](../tools/op-atk-adapt/SKILL.md#atk-跑前环境基线权威本仓库唯一展开点) 一致：

```bash
unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2
```

双标杆 **GPU 侧** `ATK_BIND_CPU_TYPE` 以团队约定为准（常见 `1` 或 `2`），**两端不要混用未记录的取值**。

---

## 4. 远端 GPU 脚本适配（用户执行）

当需要在 GPU 上跑 **内存/精度对照脚本**（非 ATK 调度）时，由用户在 GPU 环境：

1. 将 NPU 侧测试脚本 **适配为 CUDA**（`torch.npu` → `torch.cuda`，`.npu()` → `.cuda()`）  
2. 使用与 NPU  Case **相同输入文件**（优先 ATK `-sd` 产物，见 [gotchas · 输入同步](../knowledge/gotchas.md#4-gpunpu-输入不同步导致假精度问题)）  
3. 产物路径写入 `运行记录.md`；助手 **不**主动 scp/改远端（远端只读边界）

---

## 5. 常见失败签名

| 日志/现象 | 优先怀疑 |
|----------|---------|
| HTTP 404 / download gpu output | output_path 或 host/port 错误 |
| Connection refused | `atk server` 未起 / 防火墙 / 跳板未通 |
| accuracy 全 FAIL 但无 aclnn 日志 | GPU 未参与或标杆未对齐 |
