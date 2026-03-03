import torch
import torch_npu

# 设置设备
device = torch.device('npu:0')

# 测试参数
seq_len = 5
batch_size = 2
input_size = 10
hidden_size = 20
num_layers = 2
bidirectional = False
has_biases = True
batch_first = False

# 计算方向数
D = 2 if bidirectional else 1

# 构造输入张量
if batch_first:
    input_tensor = torch.randn(batch_size, seq_len, input_size, device=device)
    grad_y = torch.randn(batch_size, seq_len, D * hidden_size, device=device)
else:
    input_tensor = torch.randn(seq_len, batch_size, input_size, device=device)
    grad_y = torch.randn(seq_len, batch_size, D * hidden_size, device=device)

# 构造隐藏状态和细胞状态
h0 = torch.randn(D * num_layers, batch_size, hidden_size, device=device)
c0 = torch.randn(D * num_layers, batch_size, hidden_size, device=device)
hx = [h0, c0]

# 构造参数
# LSTM参数: 每个层有4个门，每个门有输入权重和隐藏权重，可选偏置
# 权重形状: (4 * hidden_size, input_size) 或 (4 * hidden_size, hidden_size)
# 偏置形状: (4 * hidden_size)
params = []
for layer in range(num_layers):
    for direction in range(D):
        # 输入权重
        w_ih = torch.randn(4 * hidden_size, input_size if layer == 0 else hidden_size, device=device)
        params.append(w_ih)
        # 隐藏权重
        w_hh = torch.randn(4 * hidden_size, hidden_size, device=device)
        params.append(w_hh)
        if has_biases:
            # 输入偏置
            b_ih = torch.randn(4 * hidden_size, device=device)
            params.append(b_ih)
            # 隐藏偏置
            b_hh = torch.randn(4 * hidden_size, device=device)
            params.append(b_hh)

# 构造中间状态张量 (i, j, f, o, h, c, tanhc)
# 这些张量通常由前向传播生成
if batch_first:
    i = torch.randn(batch_size, seq_len, D * num_layers, hidden_size, device=device)
    j = torch.randn(batch_size, seq_len, D * num_layers, hidden_size, device=device)
    f = torch.randn(batch_size, seq_len, D * num_layers, hidden_size, device=device)
    o = torch.randn(batch_size, seq_len, D * num_layers, hidden_size, device=device)
    h = torch.randn(batch_size, seq_len, D * num_layers, hidden_size, device=device)
    c = torch.randn(batch_size, seq_len, D * num_layers, hidden_size, device=device)
    tanhc = torch.randn(batch_size, seq_len, D * num_layers, hidden_size, device=device)
else:
    i = torch.randn(seq_len, batch_size, D * num_layers, hidden_size, device=device)
    j = torch.randn(seq_len, batch_size, D * num_layers, hidden_size, device=device)
    f = torch.randn(seq_len, batch_size, D * num_layers, hidden_size, device=device)
    o = torch.randn(seq_len, batch_size, D * num_layers, hidden_size, device=device)
    h = torch.randn(seq_len, batch_size, D * num_layers, hidden_size, device=device)
    c = torch.randn(seq_len, batch_size, D * num_layers, hidden_size, device=device)
    tanhc = torch.randn(seq_len, batch_size, D * num_layers, hidden_size, device=device)

# 构造输出梯度
grad_hy = torch.randn(D * num_layers, batch_size, hidden_size, device=device)
grad_cy = torch.randn(D * num_layers, batch_size, hidden_size, device=device)

# 其他参数
dropout = 0.0
train = True
batch_sizes = None  # 用于可变长度序列，这里使用固定长度

print("测试开始...")
try:
    # 调用torch_npu._lstm_backward_npu
    output = torch_npu._lstm_backward_npu(
        grad_y,
        grad_hy,
        grad_cy,
        input_tensor,
        hx,
        params,
        i,
        j,
        f,
        o,
        h,
        c,
        tanhc,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        batch_sizes
    )
    
    # 解析输出
    grad_input, grad_h_prev, grad_c_prev, grad_params = output
    
    print("测试成功!")
    print(f"grad_input形状: {grad_input.shape}")
    print(f"grad_h_prev形状: {grad_h_prev.shape}")
    print(f"grad_c_prev形状: {grad_c_prev.shape}")
    print(f"grad_params数量: {len(grad_params)}")
    for i, param in enumerate(grad_params):
        print(f"  grad_params[{i}]形状: {param.shape}")
        
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
