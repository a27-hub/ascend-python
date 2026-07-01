import torch


def test_stft_empty_input():
    # 对应用例规格:
    # input: shape [0, 19], fp32
    # n_fft=18, hop_length=8, win_length=14
    # window: shape [14], fp32
    # center=False, pad_mode='replicate', normalized=True,
    # onesided=False, return_complex=False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    input_tensor = torch.zeros(0, 19, dtype=torch.float32, device=device)
    window = torch.ones(14, dtype=torch.float32, device=device)

    output = torch.stft(
        input_tensor,
        n_fft=18,
        hop_length=8,
        win_length=14,
        window=window,
        center=False,
        pad_mode="replicate",
        normalized=True,
        onesided=False,
        return_complex=False,
    )

    print(f"input shape: {tuple(input_tensor.shape)}")
    print(f"output shape: {tuple(output.shape)}")
    print(f"output numel: {output.numel()}")

    expected_shape = (0, 18, 1, 2)
    assert tuple(output.shape) == expected_shape, (
        f"shape mismatch: got {tuple(output.shape)}, expected {expected_shape}"
    )
    assert output.numel() == 0, f"numel should be 0, got {output.numel()}"
    print("PASS: output shape matches expected [0, 18, 1, 2]")


if __name__ == "__main__":
    test_stft_empty_input()
