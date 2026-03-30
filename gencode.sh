    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_fp8_bfloat16_output(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 4
        win = 4
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.bfloat16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randint(-1, 1, (cout,), dtype=torch.int32)

        offset = None
        input_dtype = None
        weight_dtype = None

        conv_input_npu = conv_input.to(torch.float8_e4m3fn).npu()
        weight_npu = weight.to(torch.float8_e4m3fn).npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias, cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_fp8_float32_output(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 4
        win = 4
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float32

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randint(-1, 1, (cout,), dtype=torch.int32)

        offset = None
        input_dtype = None
        weight_dtype = None

        conv_input_npu = conv_input.to(torch.float8_e4m3fn).npu()
        weight_npu = weight.to(torch.float8_e4m3fn).npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias, cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_fp8_float32_bias(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 4
        win = 4
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randn((cout,), dtype=torch.float32)

        offset = None
        input_dtype = None
        weight_dtype = None

        conv_input_npu = conv_input.to(torch.float8_e4m3fn).npu()
        weight_npu = weight.to(torch.float8_e4m3fn).npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias.to(torch.int32), cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_hifp8_float16_output(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 4
        win = 4
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randn((cout,), dtype=torch.float32)

        offset = None
        input_dtype = torch_npu.hifloat8
        weight_dtype = torch_npu.hifloat8

        conv_input_npu = conv_input.npu()
        weight_npu = weight.npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias.to(torch.int32), cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_hifp8_bfloat16_output(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 4
        win = 4
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.bfloat16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randn((cout,), dtype=torch.float32)

        offset = None
        input_dtype = torch_npu.hifloat8
        weight_dtype = torch_npu.hifloat8

        conv_input_npu = conv_input.npu()
        weight_npu = weight.npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias.to(torch.int32), cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_hifp8_float32_output(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 4
        win = 4
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float32

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randn((cout,), dtype=torch.float32)

        offset = None
        input_dtype = torch_npu.hifloat8
        weight_dtype = torch_npu.hifloat8

        conv_input_npu = conv_input.npu()
        weight_npu = weight.npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias.to(torch.int32), cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_hifp8_hifloat8_output(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 4
        win = 4
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch_npu.hifloat8

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randn((cout,), dtype=torch.float32)

        offset = None
        input_dtype = torch_npu.hifloat8
        weight_dtype = torch_npu.hifloat8

        conv_input_npu = conv_input.npu()
        weight_npu = weight.npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias.to(torch.int32), cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_fp8_stride2(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 8
        win = 8
        cout = 1
        kh = 3
        kw = 3
        stride = (2,2)
        padding = (1,1)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randint(-1, 1, (cout,), dtype=torch.int32)

        offset = None
        input_dtype = None
        weight_dtype = None

        conv_input_npu = conv_input.to(torch.float8_e4m3fn).npu()
        weight_npu = weight.to(torch.float8_e4m3fn).npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias, cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_hifp8_stride2(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 8
        win = 8
        cout = 1
        kh = 3
        kw = 3
        stride = (2,2)
        padding = (1,1)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randn((cout,), dtype=torch.float32)

        offset = None
        input_dtype = torch_npu.hifloat8
        weight_dtype = torch_npu.hifloat8

        conv_input_npu = conv_input.npu()
        weight_npu = weight.npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias.to(torch.int32), cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_fp8_dilation2(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 8
        win = 8
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (2,2)
        dilation = (2,2)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randint(-1, 1, (cout,), dtype=torch.int32)

        offset = None
        input_dtype = None
        weight_dtype = None

        conv_input_npu = conv_input.to(torch.float8_e4m3fn).npu()
        weight_npu = weight.to(torch.float8_e4m3fn).npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias, cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_hifp8_dilation2(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 8
        win = 8
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (2,2)
        dilation = (2,2)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = torch.randn((cout,), dtype=torch.float32)

        offset = None
        input_dtype = torch_npu.hifloat8
        weight_dtype = torch_npu.hifloat8

        conv_input_npu = conv_input.npu()
        weight_npu = weight.npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, bias.to(torch.int32), cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = bias.npu() if bias else None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_fp8_no_bias(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 4
        win = 4
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = None

        offset = None
        input_dtype = None
        weight_dtype = None

        conv_input_npu = conv_input.to(torch.float8_e4m3fn).npu()
        weight_npu = weight.to(torch.float8_e4m3fn).npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, torch.zeros((cout,), dtype=torch.int32), cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)

    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_conv2d_hifp8_no_bias(self):
        torch.manual_seed(0)
        batch = 1
        cin = 1
        hin = 4
        win = 4
        cout = 1
        kh = 3
        kw = 3
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)
        groups = 1
        offset_x = 0
        round_mode = "round"
        output_dtype = torch.float16

        conv_input = torch.randint(-1, 1, (batch, cin, hin, win), dtype=torch.int8)
        weight = torch.randint(-1, 1, (cout, int(cin/groups), kh, kw), dtype=torch.int8)
        bias = None

        offset = None
        input_dtype = torch_npu.hifloat8
        weight_dtype = torch_npu.hifloat8

        conv_input_npu = conv_input.npu()
        weight_npu = weight.npu()
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, torch.zeros((cout,), dtype=torch.int32), cout, stride, padding, dilation, groups)
        scale_npu = scale.npu()
        bias_npu = None
        custom_output = torch_npu.npu_quant_conv2d(
            conv_input_npu, weight_npu, scale_npu, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias_npu, offset, input_dtype, weight_dtype)