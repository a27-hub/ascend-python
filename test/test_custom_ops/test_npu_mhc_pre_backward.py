import torch
import torch_npu
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuMhcPre(TestCase):

    def mhc_forward_pre(self, x, phi, alpha, bias, gamma, outflag: bool=False, norm_eps: float=1e-6, hc_eps: float=1e-6):

        # ---- shape ----
        B, S, n, D = x.shape
        nD = n * D

        # ---- flatten ----
        # [B, S, nD]
        vecX = x.reshape(B, S, nD).float()


        # ----gamma加权后再做GEMM ----
        # [B, S, n^2 + 2n]
        h_mix = torch.matmul(vecX * gamma.reshape(1, 1, nD), phi.t())

        # ---- RMSNorm (delayed) ----
        # [B, S, 1]
        inv_rms = torch.rsqrt(vecX.square().mean(-1, keepdim=True) + norm_eps)
        # [B, S, n^2 + 2n]
        h_mix_tmp = h_mix * inv_rms
        # [B, S, n], [B, S, n], [B, S, n*n]
        h_pre1, h_post1, h_res1 = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
        # [B, S, n, n]
        h_res2 = h_res1.reshape(B, S, n, n)
        a_pre, a_post, a_res = alpha

        # ---- split ----
        h_pre2  = a_pre * h_pre1    + bias[:n]                 # [B, S, n]
        h_post2 = a_post * h_post1   + bias[n:2*n]              # [B, S, n]
        h_res  = a_res   * h_res2    + bias[2*n:].view(n, n)    # [B, S, n, n]

        # ---- nonlinear ----
        h_pre  = torch.sigmoid(h_pre2) + hc_eps                 # [B, S, n]
        h_post = 2.0 * torch.sigmoid(h_post2)                   # [B, S, n]

        # ---- h_in = h_pre @ x ----
        # [B, S, D] = [B, S, n, 1] * [B, S, n, D]
        h_in_fp = (h_pre.unsqueeze(-1) * x.float()).sum(dim=2)
        h_in = h_in_fp.to(torch.bfloat16)

        if not outflag:
            # [B, S, D],  [B, S, n],  [B, S, n, n]
            return h_in, h_post, h_res 
        else:
            inv_rms = inv_rms.squeeze(-1) # [B, S]
            # ..., [B, S],  [B, S, n^2 + 2n],  [B, S, n]
            return h_in, h_post, h_res, inv_rms, h_mix, h_pre

    # 手动反向传播
    def cpu_op_exec(self,
        x, phi, alpha, bias,
        inv_rms, h_mix, h_pre, h_post,
        dh_in, dh_post, dh_res, gamma, 
        norm_eps: float=1e-6, hc_eps: float=1e-6, grad_x_post=None
    ):
        B, S, n, D = x.shape
        nD = n * D
        vecX = x.reshape(B, S, nD).float()
        a_pre, a_post, a_res = alpha

        dx = torch.zeros_like(x)
        dphi = torch.zeros_like(phi)
        dalpha = torch.zeros_like(alpha)
        dbias = torch.zeros_like(bias)

        # ========================================================
        # ---- 计算导数 ----
        # (V0) 反推sigmod/线性变换梯度
        # ========================================================
        x_fp = x.float() # [B, S, n, D]
        h_in_fp_grad = dh_in.float() # [B, S, D]
        dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1) # [B, S, n]
        
        s_pre2 = h_pre - hc_eps                                     # [B, S, n]   
        dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)                  # [B, S, n]
        dh_post2 = dh_post * h_post * (1.0 - h_post / 2)            # [B, S, n]

        dh_pre1 = a_pre * dh_pre2                                   # [B, S, n]
        dh_post1 = a_post * dh_post2                                # [B, S, n]
        dh_res1 = a_res * dh_res                                    # [B, S, n, n]

        dh_mix_tmp = torch.cat([dh_pre1, dh_post1, dh_res1.reshape(B, S, n * n)], dim=-1)   # [B, S, n^2 + 2n]
        dh_mix = dh_mix_tmp * inv_rms[:, :, None]                   # [B, S, n^2 + 2n]

        # ========================================================
        # (C0)
        # ========================================================
        dvecX_mm = torch.matmul(dh_mix, phi)                     # [B, S, nD]

        # ========================================================
        # (C1)
        # ========================================================
        xrs = x_fp.reshape(B*S, n*D) * gamma.reshape(1, n*D)            # [BS, nD]
        dphi = torch.matmul(dh_mix.reshape(B*S, n*n + 2*n).t(), xrs)    # [n^2 + 2n, nD]

        # ========================================================
        # (V1)
        # ========================================================
        h_mix_tmp = h_mix * inv_rms[:, :, None]                         # [B, S, n^2 + 2n]
        h_pre1, h_post1, h_res1 = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
        dinv_rms = (dh_mix_tmp * h_mix).sum(dim=-1, keepdim=True)       # [B, S, 1]

        dalpha_pre = (dh_pre2 * h_pre1).sum()
        dalpha_post = (dh_post2 * h_post1).sum()
        dalpha_res = (dh_res.reshape(B, S, n * n) * h_res1).sum()
        dalpha = torch.stack([dalpha_pre, dalpha_post, dalpha_res])

        dbias_pre = dh_pre2.sum(dim=(0, 1))                         # [n]
        dbias_post = dh_post2.sum(dim=(0, 1))                       # [n]
        dbias_res = dh_res.reshape(B, S, n * n).sum(dim=(0, 1))     # [n*n]
        dbias = torch.cat([dbias_pre, dbias_post, dbias_res], dim=0)

        # ========================================================
        # (V2)
        # ========================================================
        dvecX_inv = -(dinv_rms * inv_rms[:, :, None].pow(3) / nD) * vecX    # [B, S, nD]
        dvecX_hin = h_pre.unsqueeze(-1) * dh_in.unsqueeze(2)                # [B, S, n, D]
        dvecX_inv = dvecX_inv.reshape(B, S, n, D) + dvecX_hin               # [B, S, n, D]

        dgamma = (x_fp.reshape(B * S, n * D) * dvecX_mm.reshape(B * S, n * D)).sum(dim=-2)
        dx = dvecX_mm.reshape(B, S, n, D) * gamma.reshape(1, 1, n, D) + dvecX_inv
        if grad_x_post != None:
            dx = dx + grad_x_post.reshape(B, S, n, D).float()
        return dx.to(torch.bfloat16), dphi, dalpha, dbias, dgamma   

    def npu_op_exec(self, x, phi, alpha,
            dh_in, dh_post, dh_res,
            inv_rms, h_mix, h_pre, h_post, 
            gamma, hc_eps, grad_x_post=None):
        return torch_npu.npu_mhc_pre_backward(
            x, phi, alpha, dh_in, dh_post, dh_res,
            inv_rms, h_mix, h_pre, h_post, 
            gamma=gamma, hc_eps=hc_eps, grad_x_post=grad_x_post
        )

    def build_input_tensors(self, B, S, n, D, with_grad_x_post=False):
        x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
        phi = torch.randn(n * n + 2 * n, n * D, dtype=torch.float32)
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1
        gamma = torch.randn(n, D, dtype=torch.float32)

        dh_in = torch.randn(B, S, D).bfloat16()
        dh_post = torch.randn(B, S, n)
        dh_res = torch.randn(B, S, n, n)

       
        # 正向传播
        h_in, h_post, h_res, inv_rms, h_mix, h_pre = self.mhc_forward_pre(
            x, phi, alpha, bias, gamma, outflag=True
        )

        grad_x_post = None
        if with_grad_x_post:
            grad_x_post = torch.randn(B, S, n, D, dtype=torch.bfloat16)

        return (x, phi, alpha, bias, gamma,
                dh_in, dh_post, dh_res, inv_rms, h_mix, h_pre, h_post, grad_x_post)

    def run_and_check(self, B, S, n, D, with_grad_x_post=False):
        with torch.no_grad():
            (x, phi, alpha, bias, gamma,
                dh_in, dh_post, dh_res, inv_rms, h_mix, h_pre, h_post,
                grad_x_post) = self.build_input_tensors(B, S, n, D, with_grad_x_post)

            expected_output = self.cpu_op_exec(x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
                dh_in, dh_post, dh_res, gamma, hc_eps=1e-6, grad_x_post=grad_x_post)
            
            npu_args = [t.npu() for t in [ x, phi, alpha, dh_in, dh_post, dh_res,
                                        inv_rms, h_mix, h_pre, h_post]]
            actual_output = self.npu_op_exec(
                *npu_args, 
                gamma=gamma.npu(),
                hc_eps=1e-6,
                grad_x_post=grad_x_post.npu() if grad_x_post is not None else None,
            )

            output_names = ["dx", "dphi", "dalpha", "dbias", "dgamma"]
            tol_map = {
                "dx": 2 ** -7,
                "dphi": 1e-3,
                "dalpha": 1e-3,
                "dbias": 1e-3,
                "dgamma": 1e-3,
            }

            for name, exp, act in zip(output_names, expected_output, actual_output):
                act_cpu = act.float().cpu()
                if name == "dgamma":
                    act_cpu = act_cpu.reshape(exp.shape)
                self.assertRtolEqual(exp.float().numpy(), act_cpu.numpy(), prec=tol_map[name])

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_without_grad_x_post(self, device="npu"):
        self.run_and_check(B=1, S=1024, n=4, D=5120)

    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_pre_backward_with_grad_x_post(self, device="npu"):
        self.run_and_check(B=1, S=1024, n=4, D=5120, with_grad_x_post=True)


if __name__ == "__main__":
    run_tests()
