# Test Coverage Requirements

## Required Scenarios

1. **Basic functionality**: Verify the operator's core function works correctly
2. **Data type coverage**: Test all supported dtypes (float16, bfloat16, int8, etc.)
3. **Dimension coverage**: Test different input dimensions (min, max, typical)
4. **Optional parameter coverage**: Test all optional parameters with different values
5. **Boundary conditions**: Test edge cases (min/max values, empty tensors, etc.)

## Operator-Specific Coverage

### Quantization Operators

- Test all quant_mode values (pertoken/perchannel/pertensor)
- Test all dst_type values (INT8/INT4/FLOAT8, etc.)
- Test smooth_scales parameter
- Test group_index parameter (MoE scenario)
- Verify scale shape is correct (varies by quant_mode)

### Attention Operators

- Test all input_layout values (BSH/BNSD/BSND/TND/NTD, etc.)
- Test different scenarios (PFA/IFA/PageAttention)
- Test atten_mask parameter
- Test actual_seq_lengths parameter
- Test block_table parameter (PageAttention)
- Test quantization-related parameters
- Verify softmax_lse output (if supported)

### Matrix Operation Operators

- Test all input/output combinations (multi-multi-multi / single-multi-single / single-multi-multi / multi-multi-single)
- Test all split_item values
- Test bias parameter
- Test quantization-related parameters (scale/offset/antiquant_scale, etc.)
- Test group_list parameter
- Verify output shape is correct

### Activation Operators

- Test all supported data types
- Test in-place and out-of-place modes (if supported by operator)
- Test boundary input values (max/min/zero)
- Verify output shape matches input shape

### Normalization Operators

- Test all supported data types
- Test epsilon parameter with different values (default, very small, relatively large)
- Test with bias and without bias modes
- Test different dimensions (1D/2D/3D normalization)
- Verify auxiliary outputs (rstd/mean) have correct shapes

## Special Scenario Mapping Table

Determine required scenarios from CHECKLIST.md **Operator Classification**:

| Operator Classification | Required Scenarios |
|-------------------------|-------------------|
| Quantization | All quant_mode values, all dst_type values, smooth_scales, group_index (MoE) |
| Attention | All input_layout values, PFA/IFA/PageAttention, atten_mask, actual_seq_lengths, block_table |
| Matrix ops | All input/output combos, all split_item values, bias, quantization params, group_list |
| Activation | All supported dtypes, in-place/out-of-place (if supported), boundary input values |
| Normalization | All supported dtypes, epsilon value range, with/without bias, different dimensions |
| INT4/dst_type | Each dst_type tested individually, INT4 shape verification |

## Parameter Combination Coverage

"At least 1 per parameter" is the minimum condition. It does not guarantee combination coverage (e.g., quant_mode×dst_type minimum coverage may miss interaction combos). Rules:

- **Parameters with interactions → must be combined**. Read interaction relationships from the `Interaction: <A>×<B>` anchor in the CHECKLIST.md **Constraints** column (Step 2 model annotates in natural language, e.g., `Interaction: quant_mode×dst_type, INT8→output int8; INT4→last dim must be divisible by 2`). Step 3 model adds one test row per value branch, named `test_<A>_<B>_<value>`.
- **Orthogonal parameters → cover independently** (at least 1 each, no forced combination)
- No pairwise framework introduced; interaction pairs are explicitly marked by `Interaction:` anchor in the **Constraints** column. No annotation = default orthogonal.
- dst_type Attr driving output dtype interactions are already auto-covered by the baseline matrix (`has_dst_type` logic). Model should not duplicate generation.

## Step 2: Constraints 列填写规范

Step 2 reads the `.md` operator doc and fills the pending columns in CHECKLIST.md `### Input/Output Spec` tables — the "Constraints" column in the input table and the "Shape Rule" column in the output table.

> 🔒 **Never modify machine-written columns**: The input table columns "Type" and "Supported dtypes", plus the output table column "Dtype Rule" — these three columns are machine-written by step 1 scaffold_init. Format: `Tensor` / `int (Attr)`, dtypes separated by `, `, dtype rule as enum values. Step 3 scripts parse this exact format. **The model may ONLY fill the "Constraints" column and the output "Shape Rule" column. Never modify the three machine-written columns.** Changing `Tensor`→`Tensor?`, `, `→`; `, or dtype enum→natural language prose will cause step 3 parsers to collapse multi-values and produce zero parameter coverage. Express "nullable", "default value", or "dtype linkage" entirely in the "Constraints" / "Shape Rule" columns.

### Input Table "Constraints" Column — Fill in Natural Language

Read the following sections of `.md`. Scan each parameter against the checklist. Fill the constraints column in **natural language**. Separate multiple constraints with `;`.

**Structured Reading Checklist** (per parameter, check `.md` in order):

A. Parameter table "Usage Notes" column →
   - Shape constraints? ("shape is [H]", "last dim must be multiple of 2")
   - Support / non-support conditions? ("optional, supports null pointer", "when X then shape is Y")

B. "Constraints" section →
   - Cross-parameter conditions? ("when X is INT32, Y cannot be null")
   - Value ranges? ("groupNum >= 1")
   - ⚠️ **Platform-specific restrictions do NOT go here** (see "Platform Restriction Exclusion" below)

C. "Return Value / Error Code" section →
   - Parameter validity rules? ("dtype not in supported range" → implies dtype constraint)

D. **Cross-parameter interaction check (required)** — Does the value of parameter A affect the behavior/shape of parameter B?
   - Yes → **Must** write an `Interaction: <A>×<B>, <value rule>` anchor. The value rule must **list enumerable discrete branches** in format `value1→consequence1; value2→consequence2`. Use `;` to separate branches. Step 4 uses these branches to generate separate test methods.
     Example: `Interaction: dst_type×y, INT8→output int8; INT4→last dim must be divisible by 2` (step 4 generates `test_dst_type_y_int8` / `test_dst_type_y_int4`)
   - No → Skip

For each constraint found, write one natural-language sentence. If confirmed no constraint exists, write `None` (to distinguish "confirmed none" from "missed").

> ⚠️ **Platform Restriction Exclusion**: Constraints tagged with platform qualifiers (`<term>Atlas …</term>`, `<term>Ascend …</term>`, "950PR only…", "Atlas Inference Series…") do **not** go in the constraints column. They are platform-conditional constraints. Leave them to step 3 handling per target platform (910B).
> Test: does the restriction still hold after removing the platform qualifier? "x only FLOAT16" (Atlas Inference Series) — after removal, it does not hold (910B supports FLOAT16+BFLOAT16 for x) → platform-conditional → step 3. **Never** put it in the constraints column. Doing so would cause step 4 to wrongly delete valid 910B test cases.
> Only parameter-inherent constraints that "hold on all platforms" go in the constraints column (e.g., "smooth_scales dtype matches x").

### Output Table "Shape Rule" Column

Read `.md` output parameter dimension column. Format: `same as x.shape` / `[batch, hidden]` / `last dim = x last dim / 8`.

## Step 3: 覆盖矩阵终定规范

The script produces only the baseline (mechanical enumeration). The model reads the "Constraints" column and reshapes the matrix into its final form. All additions and deletions below happen in this step. Step 4 receives the finalized matrix and only writes code.

### Add/Delete Actions (4 types)

1. **Platform exception row deletion**: Read the `.md` platform support table. Delete rows that are exceptions for the target platform 910B (e.g., "Atlas Inference Series: x only FLOAT16" → delete bf16 rows). UT targets 910B. 910B usually supports the broadest set → exceptions are rare. Most cases need no action. If no exceptions: note "No platform exceptions" in CHECKLIST.
2. **Not-applicable row deletion**: Remove **all rows** containing that parameter (including product combination rows). Example: "group_index is not applicable" → delete all rows containing group_index, not just single-parameter rows.
3. **Interaction expansion row addition**: Read the `Interaction: <A>×<B>` anchor in the "Constraints" column. Parse the branch value descriptions. Add one test row per branch. Name it `test_<A>_<B>_<value>`. ⚠️ First check if the baseline already has it (e.g., dst_type type A is already generated). Skip if it exists. Avoid collisions.
4. **Conditional omission** (e.g., "per-tensor mode: do not pass smooth_scales") → Annotate that combination to skip the parameter. Or convert to negative test case.

When uncertain: annotate and continue. Do not block.

### 🛡️ Anti-overdeletion guardrail (step 3 finalization hard constraint)

The script baseline is the coverage floor, not a draft.

- **Positive row count must not drop below baseline**: Final matrix positive row count `≥` script "Auto-generated stats: N test items". Delete rows only for three explicit reasons: platform exception / parameter not applicable / fully covered by another existing row.
- **Every deleted row needs a reason recorded**: Append a `### Finalization Add/Delete Log` section after the CHECKLIST "Coverage Matrix" section. Write one line per change using the exact format contract: `Delete <exact_method_name>: <reason>` or `Add <exact_method_name>: <reason>`. `<exact_method_name>` **MUST be the verbatim Test Item column value (逐字方法名)** from the Coverage Matrix or Negative Coverage Matrix — natural-language summaries (e.g. "Delete fp8 三项") are forbidden. `reconcile.py` relies on this format for deletion pass-through; format violations produce `[BAD-LOG]` warnings. Unreasoned deletion is a defect.
- **Required-scenario check must not regress**: Scenarios marked ✅ in the script "Required scenario check" must remain ✅ after finalization. Scenarios marked ⚠️ missing must have rows added or have a note in the add/delete log stating "not applicable".
- **Degradation banner must be resolved**: If the self-check area shows a "Negative degradation" warning, re-run step 1 scaffold_init to generate `.constraint_info.json`, then return to step 3. Do not carry a degraded matrix into step 4.

## Constraint-Driven Data Construction

The **Constraints** column contains **natural language** constraints (script pre-fill + model supplement). Step 4 model reads it to construct legal inputs:

- Cross-parameter constraints (e.g., "smooth_scales last dim = x last dim", "per-tensor mode: scale is 1D") → Step 4 reads natural language description, constructs constraint-satisfying inputs
- Interaction combination values are in the natural language description after the `Interaction: <A>×<B>` anchor in the **Constraints** column. Step 3 expands each value into N test rows. Step 4 writes code per the matrix row by row.
- The **Constraints** column must be fully filled in Step 2. Otherwise Step 4 may construct data violating preconditions, triggering ACL errors instead of valid tests.
- Row add/delete (not-applicable deletion, interaction expansion addition, platform exception deletion) is handled by Step 3 model. The script does not parse the Constraints column.

## Reference Implementation Coverage

The reference implementation (`supported_op_exec` for custom ops, `cpu_op_exec` for native ops) is the **gold standard** for every test case. It MUST be able to produce a correct golden output for every matrix row. Gaps in reference coverage produce fake tests (shape-only assertions without value verification) and are defects.

### Requirements

1. **Every output dtype branch must have a working reference path**: If the Coverage Matrix includes `test_dst_type_int4`, the reference MUST correctly compute INT4 output (including quantization + packing logic). If the reference cannot handle a dtype → either extend the reference to support it, or delete the row with a logged reason (`Delete <item>: 参考实现不支持<dtype>` in `### Finalization Add/Delete Log`).

2. **Every optional parameter combination must have a working reference path**: If the matrix has `test_smooth_scales_group_index_fp16`, the reference MUST correctly process both `smooth_scales` and `group_index` together. Parameter interactions that crash the reference (e.g. `group_index` requires `smooth_scales` but `smooth_scales=None` is passed) → the reference is incomplete. Either fix it or delete the row.

3. **Multi-scenario-test (MST) rows must all resolve**: If MST (multi-scenario-test) rows exist in the Coverage Matrix, the reference MUST produce valid output for **every** scenario variant. A reference that handles only one variant of a multi-scenario operator leaves the other variants as fake tests → either extend the reference to cover all variants, or Delete-log the uncovered variants.

4. **No shape-only smoke tests disguised as value tests**: A test method that calls `supported_op_exec` but only asserts output shape/dtype without comparing golden values is a **fake test**. Every golden value computed by the reference MUST be consumed by `assertRtolEqual`. See `test-template.md` Assertion Completeness Iron Rule for per-dtype rules.

5. **Pre-step-4 gate**: Before translating the matrix into code (step 4), scan the finalized matrix rows against the reference implementation. For each row ask: "Can the reference compute a correct golden for this parameter/dtype combination?" If no → fix the reference or add a Delete log entry. Do not carry unsupported rows into code generation.

### Common Gaps

| Gap | Symptom | Fix |
|-----|---------|-----|
| Reference only handles int8/int4 | fp8/hf8 rows exist but reference returns wrong dtype or crashes | Extend reference or Delete-log fp8 rows |
| Reference crashes on parameter combination | `group_index` passed without `smooth_scales` → IndexError | Add None-guard in reference or Delete-log row |
| Reference computes golden but test only asserts shape | `assertEqual(output.shape, ...)` without `assertRtolEqual(cpu, npu)` | Fix test to compare values; if reference golden is unreliable, fix reference first |

## Test Method Naming Conventions

```python
def test_basic_functionality(self):          # basic function
def test_dtype_float16(self):                # data type
def test_dtype_bfloat16(self):
def test_with_optional_param(self):          # optional parameter
def test_quant_mode_pertoken(self):          # specific mode
def test_quant_mode_perchannel(self):
def test_quant_mode_pertensor(self):
def test_moe_scenario(self):                 # specific scenario
def test_page_attention(self):
def test_min_dimension(self):                # boundary test
def test_max_dimension(self):
def test_neg_invalid_dimension(self):          # negative: invalid dim
def test_neg_invalid_dtype(self):              # negative: invalid dtype
```

## Assertion Standards

```python
# Primary assertion: relative error comparison (preferred)
self.assertRtolEqual(expected, actual, 0.001)         # fp16
self.assertRtolEqual(expected, actual, 0.0001)        # fp32
self.assertRtolEqual(a, b, prec=1e-3, prec16=1e-3)   # specify precision separately

# Shape assertion
self.assertEqual(output.shape, expected_shape)
self.assertEqual(output.shape, torch.Size([2, 32, 256]))

# Dtype assertion
self.assertEqual(output.dtype, torch.int8)

# Non-null assertion
self.assertIsNotNone(output)
self.assertTrue(output.numel() > 0)

# Range assertion (quantization operators)
self.assertTrue(torch.all(output >= -128))
self.assertTrue(torch.all(output <= 127))
```

## Error Handling Tests (Negative, migrated to `test_neg_` prefix)

Negative test cases are now unified under the `test_neg_` prefix, auto-generated by `generate_negative_cases()`. See `test-template.md` negative test spec.

```python
@unittest.skipIf(not torch.npu.is_available(), "NPU required for negative validation")
def test_neg_invalid_dimension(self, device="npu"):
    """Negative-dim out of bounds: dimension exceeds legal range, expect validation error"""
    x = torch.randn((256,), dtype=torch.float16).npu()
    with self.assertRaisesRegex(RuntimeError, r"PARAM_INVALID"):
        torch_npu.npu_operator(x)
```

## INT4 / dst_type Scenario Coverage

When the operator has a `dst_type` parameter or the _def.cpp Output DataType array contains `DT_INT4`:

1. **Generate one test case per supported output dtype**: Each value of `dst_type` (e.g., INT8, INT4, FLOAT8) needs its own test
2. **INT4-specific checks**: INT4 output is packed as INT32, so `output.shape[-1] == input.shape[-1] / 8`. Verify this shape transformation
3. **dtype assertion**: Each test must assert the output dtype matches the expected dtype for the given `dst_type` value

## Test Data Scale Guidelines

Use data sizes that exercise real-world scenarios, not just minimal tensors:

| Operator Category | Recommended Scale | Minimum |
|-------------------|-------------------|---------|
| Quantization | batch ≥ 1024, hidden ≥ 512, groups ≥ 4 | 2×2 |
| Attention | B ≥ 2, S ≥ 128, H ≥ 8 | 1×64×8 |
| Matrix ops | M,N,K ≥ 256 | 16×16 |
| Activation | elements ≥ 4096 | 128 |

Each operator should include at least one test case at the recommended scale, plus boundary tests at minimum legal sizes.

## Troubleshooting

**Test fails with shape mismatch**
- INT4 output packed as INT32: `last_dim = input_last_dim / 8`
- Quantization scale shape varies by `quant_mode`
