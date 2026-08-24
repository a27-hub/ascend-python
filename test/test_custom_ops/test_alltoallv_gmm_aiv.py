import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_inputs(world_size=2, local_experts=2, hidden=128, output=128):
    generator = torch.Generator().manual_seed(20260819)
    counts = torch.ones(
        (world_size, world_size, local_experts), dtype=torch.int64
    )
    tokens_per_rank = counts[0].sum().item()
    gmm_x = [
        torch.randn(tokens_per_rank, hidden, generator=generator).to(torch.bfloat16)
        for _ in range(world_size)
    ]
    gmm_weight = [
        torch.randn(
            local_experts, hidden, output, generator=generator
        ).to(torch.bfloat16)
        for _ in range(world_size)
    ]
    mm_x = [
        torch.randn(2, hidden, generator=generator).to(torch.bfloat16)
        for _ in range(world_size)
    ]
    mm_weight = [
        torch.randn(hidden, 64, generator=generator).to(torch.bfloat16)
        for _ in range(world_size)
    ]
    return counts, gmm_x, gmm_weight, mm_x, mm_weight


def _reference_for_rank(rank, counts, gmm_x, gmm_weight, mm_x, mm_weight):
    world_size = counts.size(0)
    local_experts = counts.size(2)
    permute_parts = []
    gmm_parts = []
    for expert in range(local_experts):
        expert_parts = []
        for source in range(world_size):
            flat_counts = counts[source].reshape(-1)
            flat_index = rank * local_experts + expert
            offset = flat_counts[:flat_index].sum().item()
            length = flat_counts[flat_index].item()
            expert_parts.append(gmm_x[source][offset:offset + length])
        expert_tokens = torch.cat(expert_parts, dim=0)
        permute_parts.append(expert_tokens)
        gmm_parts.append(
            torch.matmul(
                expert_tokens.float(), gmm_weight[rank][expert].float()
            )
        )
    return (
        torch.cat(gmm_parts, dim=0),
        torch.matmul(mm_x[rank].float(), mm_weight[rank].float()),
        torch.cat(permute_parts, dim=0),
    )


def _run_aiv_rank(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    try:
        process_group = dist.distributed_c10d._get_default_group()
        hcom = process_group._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(rank)
        counts, gmm_x, gmm_weight, mm_x, mm_weight = _make_inputs(world_size)
        golden_gmm, golden_mm, golden_permute = _reference_for_rank(
            rank, counts, gmm_x, gmm_weight, mm_x, mm_weight
        )
        send_counts = counts[rank].reshape(-1).tolist()
        recv_counts = counts[:, rank, :].reshape(-1).tolist()
        actual_gmm, actual_mm, actual_permute = torch_npu.npu_alltoallv_gmm(
            gmm_x=gmm_x[rank].npu(),
            gmm_weight=gmm_weight[rank].npu(),
            hcom=hcom,
            ep_world_size=world_size,
            send_counts=send_counts,
            recv_counts=recv_counts,
            mm_x=mm_x[rank].npu(),
            mm_weight=mm_weight[rank].npu(),
            permute_out_flag=True,
            comm_mode="aiv",
        )
        torch_npu.npu.synchronize()
        torch.testing.assert_close(
            actual_gmm.cpu().float(), golden_gmm, rtol=0.01, atol=0.01
        )
        torch.testing.assert_close(
            actual_mm.cpu().float(), golden_mm, rtol=0.01, atol=0.01
        )
        torch.testing.assert_close(
            actual_permute.cpu(), golden_permute, rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


def _run_invalid_comm_mode_rank(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    try:
        process_group = dist.distributed_c10d._get_default_group()
        hcom = process_group._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(rank)
        counts, gmm_x, gmm_weight, _, _ = _make_inputs(world_size)
        send_counts = counts[rank].reshape(-1).tolist()
        recv_counts = counts[:, rank, :].reshape(-1).tolist()
        try:
            torch_npu.npu_alltoallv_gmm(
                gmm_x=gmm_x[rank].npu(),
                gmm_weight=gmm_weight[rank].npu(),
                hcom=hcom,
                ep_world_size=world_size,
                send_counts=send_counts,
                recv_counts=recv_counts,
                comm_mode="ai_cpu",
            )
        except RuntimeError:
            return
        raise AssertionError("V2 must reject comm_mode other than 'aiv'")
    finally:
        dist.destroy_process_group()


class TestAlltoAllvGmmAiv(unittest.TestCase):
    def test_schema_exposes_comm_mode(self):
        schema = str(torch.ops.npu.npu_alltoallv_gmm.default._schema)
        self.assertIn("str? comm_mode=None", schema)

    @unittest.skipUnless(torch_npu.npu.device_count() >= 2, "requires two NPUs")
    def test_bf16_aiv_eager_two_rank(self):
        mp.spawn(_run_aiv_rank, args=(2, _get_free_port()), nprocs=2, join=True)

    @unittest.skipUnless(torch_npu.npu.device_count() >= 2, "requires two NPUs")
    def test_invalid_comm_mode_is_rejected(self):
        mp.spawn(
            _run_invalid_comm_mode_rank,
            args=(2, _get_free_port()),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
