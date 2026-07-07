# Only_Doc_Commit_Check

检查 PR 是否仅包含文档变更的 GitCode Action 插件。

## 工作原理

调用 [only_doc.py](D:\codearts\cie\plugin\only_doc.py) 执行以下流程：

1. **Pre-merge 模拟**: clone 目标仓库 → fetch PR 的 merge-request head → git merge 到目标分支
2. **变更文件分析**: `git diff-tree` 获取 PR 相对于目标分支的所有变更文件
3. **文档判断**: 检查所有变更文件后缀是否均为文档类型（`.md` / `.rst` / `.log` / `.png` / `.jpg` / `.doc`）
4. **输出结果**: JSON 文件 `{"is_only_doc": "yes"}` 或 `{"is_only_doc": "no"}`

## 输入参数

| 参数 | 必填 | 默认值 | 描述 |
|------|------|--------|------|
| `pr_id` | 是 | - | PR/MR ID，用于 fetch merge-request head 并执行预合并 |
| `target_branch` | 是 | - | 目标分支名称，作为 clone 和 merge 的基准 |
| `remote_url` | 是 | - | 目标仓库的远程 URL，Python 脚本将 clone 此仓库以执行分析 |
| `result_msg_path` | 否 | `""` | 结果消息输出目录路径 |
| `step_id` | 否 | `""` | 步骤ID，用于拼接结果文件名 `<step_id>_result.json` |

## 使用示例

```yaml
- name: Only Doc Commit Check
  uses: Only_Doc_Commit_Check
  with:
    pr_id: ${{ inputs.pr_id }}
    target_branch: ${{ inputs.TARGET_BRANCH }}
    remote_url: ${{ test_REPO_URL }}
    result_msg_path: ${{ RESULT_MSG_PATH }}
    step_id: ${{ STEP_ID }}
```

## 输出

- 成功时: 生成 `<result_msg_path>/<step_id>_result.json`，内容为 `{"is_only_doc": "yes"}` 或 `{"is_only_doc": "no"}`
- 失败时: action 以非零 exit code 退出
