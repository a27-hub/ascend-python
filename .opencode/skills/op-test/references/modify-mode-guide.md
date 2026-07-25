# Modify Mode Guide

Modify mode is a high-frequency pain point. Clear criteria for "append method vs new file" + anchor-based insertion method.

## Criteria: Append Method vs New File

| Condition | Action |
|-----------|--------|
| `test_npu_{operator_name}.py` exists with unchanged operator interface, only new test scenarios needed | Append methods to existing class |
| New operator variant (e.g., v1→v2), requiring a new test class | Create new test file |

**Typical append-method scenarios**:
- `test_npu_rms_norm.py` exists, need to add `test_with_epsilon_large` method
- `test_npu_quant.py` exists, need to add `test_quant_mode_pertensor` method

**Typical new-file scenarios**:
- `test_npu_quant.py` exists, new v2 variant → create `test_npu_quant_v2.py`
- Adding a completely different operator interface → create new `test_npu_{new_operator}.py`

## Append Method: Anchor-Based Insertion (No Sentinel Comments, No Full-File Read)

LLM failure modes when appending to existing files: long file truncation, wrong class targeting, broken indentation. Solution:

- ❌ Sentinel comments (`# @@op-test-insert-point@@`): requires pre-polluting repo files, ambiguous targeting when multiple operators share the file, real repo files don't have these markers
- ✅ Anchor-based: insertion point is a deterministically locatable text anchor. Find the anchor and insert — don't rely on the model reading the full file to understand structure. "Mechanical work should not be guessed by the model."

### Anchor Location Rules

1. **File-level append anchor**: `if __name__ == "__main__":` line → insert before it
2. **Class-level append anchor**: end of the target `class Test{Name}` block, between it and the next `class ` or `if __name__`
3. Use anchor line numbers + Edit for precise insertion. **Do NOT read the full file and rewrite** (avoids truncation / corrupting existing content)
4. Do NOT implant sentinel comments

### Steps

1. 搜索目标文件中 `class Test{Name}` 和 `if __name__ == "__main__":` 的行号
2. 根据判定逻辑确定追加位置：
   - 追加新方法到已有类 → 定位类末尾（下一个 `class ` 或 `if __name__` 行之前）
   - 插入新类 → 定位 `if __name__ == "__main__":` 行之前
3. 使用 Edit 工具在锚点位置精确插入，不重写整个文件

Battle-tested: DynamicQuant integration used this method (locate `__main__` line → anchor Edit insert), without reading the full file.

## FakeTensor Test Append

Use the same anchor-based approach when appending FakeTensor tests to `test/core_tests/test_fake_tensor.py`:

1. Search for `class Test{OperatorName}` and `if __name__ == "__main__":` line numbers in the file
2. Create mode: insert complete new class before `if __name__` line
3. Modify mode:
   - Existing class with same name → append methods at class end (before next `class ` or `if __name__`)
   - No class with that name → insert new class before `if __name__` line
4. Use anchor line numbers + Edit for precise insertion. Do not read full file and rewrite.

## Mode Source

**Mode (create/modify) is read from CHECKLIST.md "Operator Basic Info → Mode" field**, written by upstream op-adaptation. op-test does not determine mode on its own.

If CHECKLIST.md does not specify a mode field, fallback to file existence check:
- No test file in target directory → create
- Test file already exists in target directory → modify

## v1/v2 Precision Comparison

In modify mode, variants calling the same aclnn must include v1 vs v2 precision comparison. See template: `patterns/v1v2-comparison.md`.
