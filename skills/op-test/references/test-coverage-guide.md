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
def test_invalid_dimension(self):            # error handling
def test_invalid_dtype(self):
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

## Error Handling Tests

```python
@SupportedDevices(['Ascend910B'])
def test_invalid_dimension(self, device="npu"):
    """Test invalid dimension input"""
    x = torch.randn((256,), dtype=torch.float16).npu()  # 1D, should raise error
    with self.assertRaises(RuntimeError):
        torch_npu.npu_operator(x)

@SupportedDevices(['Ascend910B'])
def test_invalid_dtype(self, device="npu"):
    """Test invalid data type"""
    x = torch.randn((2, 32, 256), dtype=torch.float32).npu()
    with self.assertRaises(RuntimeError):
        torch_npu.npu_operator(x)
```
