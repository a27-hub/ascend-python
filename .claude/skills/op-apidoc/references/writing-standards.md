# Writing Standards

## Basic Standards
- All body content in Chinese; product names use `<term>` tags; math formulas use LaTeX `$$...$$`
- Omit the "Formula" section if there is no formula; omit `> [!NOTE]` block if there are no notes
- Parameter order matches the function prototype; list by product when different products support different dtypes
- Single-op examples use `>>>` interactive style with real output values; graph mode uses `torch.compile` + `npu_backend`
- Omit graph mode example if the operator does not support graph mode

## Product Support Table

Include only actually supported products:

```markdown
| Product                                                      | Supported |
| ------------------------------------------------------------ | :-------: |
|<term>Atlas A3 Training Series/Atlas A3 Inference Series</term>  |    √     |
|<term>Atlas A2 Training Series/Atlas A2 Inference Series</term>  |    √     |
|<term>Atlas Inference Series</term>                              |    √     |
|<term>Atlas Training Series</term>                               |    √     |
```

## Functional Description Requirements
- **API function**: One clear sentence describing the operator's core function
- **Formula**: Use LaTeX format to clearly show the mathematical computation
- **NOTE block**: Include the following (if applicable):
  - Special scenarios supported (e.g., MoE, PageAttention, quantization)
  - Important usage restrictions or constraints
  - Performance-related notes
  - Dependencies between parameters

## Parameter Description Requirements
- **Required parameters**: Clearly mark as "required parameter", describe purpose
- **Optional parameters**: Clearly mark as "optional parameter", state default value and purpose
- **Data types**: List all supported dtypes (e.g., `float16`, `bfloat16`, `int8`)
- **Data format**: Note supported formats (usually `$ND$`)
- **Dimension requirements**: Clearly state dimension range (e.g., "2 to 8 dimensions")
- **Special notes**: For complex parameters, add detailed descriptions:
  - Enum parameters: list all valid values and their meanings
  - String parameters: list all supported string values
  - Integer parameters: state value range and meaning of special values
  - List parameters: state length limits and element requirements

## Parameter Description Examples

**Basic parameter**:
```markdown
- **x** (`Tensor`): Required parameter. Input tensor. Supported dtypes: `float16`, `bfloat16`. Supported format: $ND$. Dimension: 2–8.
```

**Different products support different dtypes** — list by product:
```markdown
- **x** (`Tensor`): Required parameter. Input tensor. Supported format: $ND$. Dimension: 2–8.
  - <term>Atlas A3 Training Series/Atlas A3 Inference Series</term>: Supported dtypes: `float16`, `bfloat16`, `int8`.
  - <term>Atlas A2 Training Series/Atlas A2 Inference Series</term>: Supported dtypes: `float16`, `bfloat16`.
```

**Enum parameter**:
```markdown
- **quant_mode** (`str`): Optional parameter. Quantization mode. Supported values:
  - `"pertoken"`: per-token quantization, scale shape is [B, S]
  - `"perchannel"`: per-channel quantization, scale shape is [D]
  - `"pertensor"`: per-tensor quantization, scale shape is [1]
  Default: `"pertoken"`.
```

**ScalarType enum parameter** (note: dst_type is `ScalarType`, not `int`):
```markdown
- **dst_type** (`ScalarType`): Optional parameter. Output data type. Supported values:
  - `torch.int8`: INT8 (default)
  - `torch.quint4x2`: INT4 (packed as INT32, last dim of output = input last dim / 8)
  - `torch.float8_e4m3fn`: FLOAT8_E4M3FN
  - `torch.float8_e5m2`: FLOAT8_E5M2
  Default: `None` (equivalent to INT8).
```

**Layout parameter**:
```markdown
- **input_layout** (`str`): Optional parameter. Data layout format. Supported values:
  - `"BSH"`: [Batch, Sequence, Hidden]
  - `"BNSD"`: [Batch, Num_heads, Sequence, Dim]
  - `"BSND"`: [Batch, Sequence, Num_heads, Dim]
  - `"TND"`: [Total_tokens, Num_heads, Dim]
  - `"NTD"`: [Num_heads, Total_tokens, Dim]
  Default: `"BSH"`.
```

**List parameter**:
```markdown
- **group_list** (`List[int]`): Optional parameter. Matmul index list in the M direction. Dtype: `int64`. Max length: 128. Default: `None`.
```

## Return Value Requirements
- Clearly state the shape calculation rule for return values
- State the dtype of return values (fixed or derived from input)
- For multiple return values, describe each one individually
- State shape changes in special cases (e.g., INT4 packing, quantization)

**Multiple return values format**:
```markdown


- **yOut** (`Tensor`): Quantized output tensor. Shape matches input `x` (for INT4, last dim = input last dim / 8). Dtype determined by `dst_type`.
- **scaleOut** (`Tensor`): Quantization scale. Dtype: `float32`. Shape determined by `quant_mode`:
  - `"pertoken"`: input shape with last dim removed
  - `"perchannel"`: last dim of input
  - `"pertensor"`: [1]
```

## Constraint Requirements
- Must include basic constraints:
  - Supported scenarios (inference/training)
  - Whether graph mode is supported
- Operator-specific constraints:
  - Input dimension limits
  - Dtype restrictions
  - Parameter value ranges
  - Dependencies between parameters
  - Special scenario restrictions (e.g., PageAttention, quantization)
- Use tables for complex constraints (e.g., scenario-specific limits)

## Table Extraction Rules

When the .md input file contains the following types of tables, they must be fully migrated to the documentation:

1. **Enumeration parameter complete value tables** (sparse_mode, inner_precise, antiquant_mode, etc.)
   - Extract all rows, not just the "common" ones
   - Preserve all columns (value, meaning, constraints)
2. **Parameter combination constraint tables** (e.g., quantization scenario restriction tables)
   - Fully migrate, annotate required/prohibited parameters for each scenario
3. **Scenario constraint tables** (e.g., PageAttention parameter requirements)
   - Fully migrate, including B/S/D dimension restrictions

**Key Principle**: Constraint information presented in table format in .md should also be presented in table format in the documentation, not degraded to text descriptions.

## Constraint Examples

**Basic constraints**:
```markdown
- This interface supports inference and training scenarios.
- This interface supports graph mode.
- Input tensor x must have 2–8 dimensions.
- When dst_type is INT4 (`torch.quint4x2`), the last dim of x must be divisible by 8.
```

**Table-form constraints**:
```markdown
## Constraints

- This interface supports inference and training scenarios.
- This interface supports graph mode.

### Supported Data Types

| Product | Input dtype | Output dtype |
|---------|-------------|--------------|
| Atlas A3 | float16, bfloat16, int8 | float16, bfloat16, int8 |
| Atlas A2 | float16, bfloat16 | float16, bfloat16 |

### Scenario Limits

| Scenario | split_item | x list length | weight list length | y list length |
|----------|-----------|---------------|-------------------|---------------|
| multi-multi-multi | 0/1 | N | N | N |
| single-multi-single | 0/1 | 1 | N | N |
| single-multi-multi | 2/3 | N | N | 1 |
| multi-multi-single | 2/3 | 1 | N | 1 |
```

## Call Example Requirements
- **Single-op mode**: Use `>>>` interactive style, show actual output (including real tensor values)
- **Graph mode**: Use complete `torch.compile` example
- Examples should cover typical usage scenarios
- For complex operators, provide multiple examples for different scenarios
- Example data should be reasonable (dimensions, dtypes, etc.)

## Special Operator Requirements

**Quantization operators**:
- In NOTE block, describe quantization modes and scenarios (including MoE scenario with group_index)
- Describe dst_type parameter in detail (type is `ScalarType`, not `int`)
- Explain scale shape calculation rules (varies by quant_mode)
- Provide examples for different quantization modes (pertoken/perchannel/pertensor)
- Provide example using smooth_scales

**Attention operators**:
- In NOTE block, describe supported scenarios (PFA/IFA/PageAttention)
- Describe all valid values for input_layout parameter
- Describe workspace usage (if applicable)
- Provide examples for different layouts
- Use tables to show dtype support by product

**Matrix operation operators**:
- In NOTE block, describe supported input/output combination modes
- Describe split_item parameter meaning in detail
- Use tables to show scenario constraints
- Provide examples for different modes (including INT4 NZ format)

