import os
import subprocess
import sys
import textwrap

from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


CHILD_SCRIPT = r"""
import os
import sys

import torch
import torch_npu


def assert_npu_tensor(tensor):
    if tensor.device.type != "npu":
        raise AssertionError("expected npu tensor, got {}".format(tensor.device))
    if tensor.numel() != 0 and not torch.isfinite(tensor.float()).all().item():
        raise AssertionError("expected finite tensor")


def run_supported_ops():
    torch.manual_seed(2026)

    normal_tensor = torch.empty((16, 16), device="npu", dtype=torch.float32)
    normal_tensor.normal_(0.0, 1.0)
    assert_npu_tensor(normal_tensor)

    randn_tensor = torch.randn((8, 8), device="npu", dtype=torch.float32)
    assert_npu_tensor(randn_tensor)

    xavier_tensor = torch.empty((8, 16), device="npu", dtype=torch.float32)
    torch.nn.init.xavier_uniform_(xavier_tensor)
    assert_npu_tensor(xavier_tensor)

    uniform_tensor = torch.empty((16, 16), device="npu", dtype=torch.float32)
    uniform_tensor.uniform_(-2.0, 3.0)
    assert_npu_tensor(uniform_tensor)
    if uniform_tensor.min().item() < -2.0 or uniform_tensor.max().item() > 3.0:
        raise AssertionError("uniform_ result is out of range")

    random_tensor = torch.empty((64,), device="npu", dtype=torch.int32)
    random_tensor.random_(0, 100)
    assert_npu_tensor(random_tensor)
    if random_tensor.min().item() < 0 or random_tensor.max().item() >= 100:
        raise AssertionError("random_(from, to) result is out of range")


def run_unsupported_ops():
    random_tensor = torch.empty((64,), device="npu", dtype=torch.int32)
    random_tensor.random_()
    assert_npu_tensor(random_tensor)

    exponential_tensor = torch.empty((128,), device="npu", dtype=torch.float32)
    exponential_tensor.exponential_()
    assert_npu_tensor(exponential_tensor)
    if exponential_tensor.min().item() <= 0.0:
        raise AssertionError("exponential_ result should be greater than zero")

    if not hasattr(torch_npu, "npu_sim_exponential_"):
        return

    sim_exponential_tensor = torch.empty((128,), device="npu", dtype=torch.float32)
    try:
        torch_npu.npu_sim_exponential_(sim_exponential_tensor, lambd=1.0)
    except RuntimeError as err:
        message = str(err).lower()
        if "npu_sim_exponential" in message and "support" in message:
            return
        raise
    assert_npu_tensor(sim_exponential_tensor)
    if sim_exponential_tensor.min().item() <= 0.0:
        raise AssertionError("npu_sim_exponential_ result should be greater than zero")


def run_dropout():
    torch.manual_seed(2026)
    input_tensor = torch.ones((16, 16), device="npu", dtype=torch.float32)
    output_tensor = torch.nn.functional.dropout(input_tensor, p=0.5, training=True)
    assert_npu_tensor(output_tensor)


def set_force_incompatible(value):
    torch_npu._C._npu_setOption({"TORCH_NPU_FORCE_INCOMPATIBLE": value})


def set_global_compatible(value):
    torch_npu.npu.use_compatible_impl(value)


def main():
    case = sys.argv[1]
    if case == "supported":
        run_supported_ops()
    elif case == "unsupported":
        run_unsupported_ops()
    elif case == "dropout":
        run_dropout()
    elif case == "set_option_before":
        set_force_incompatible("randomness")
        run_supported_ops()
    elif case == "set_option_after":
        run_supported_ops()
        set_force_incompatible("randomness")
        set_global_compatible(False)
        run_supported_ops()
    elif case == "global_switch_ignored":
        set_global_compatible(os.environ["GLOBAL_COMPATIBLE"] == "1")
        run_supported_ops()
    else:
        raise AssertionError("unknown child case: {}".format(case))


if __name__ == "__main__":
    main()
"""


class TestRandomnessCompatible(TestCase):
    def run_child_case(self, case, extra_env=None):
        env = os.environ.copy()
        env.pop("TORCH_NPU_FORCE_INCOMPATIBLE", None)
        env.pop("TORCH_NPU_USE_COMPATIBLE_IMPL", None)
        if extra_env is not None:
            env.update(extra_env)

        command = [sys.executable, "-c", CHILD_SCRIPT, case]
        result = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        message = textwrap.dedent(
            """\
            child case: {case}
            return code: {returncode}
            stdout:
            {stdout}
            stderr:
            {stderr}
            """
        ).format(
            case=case,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        self.assertEqual(result.returncode, 0, message)

    @SupportedDevices(["Ascend910B"])
    def test_a2_supported_ops_without_blacklist(self):
        self.run_child_case("supported")

    @SupportedDevices(["Ascend910B"])
    def test_a2_supported_ops_with_blacklist(self):
        self.run_child_case("supported", {"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"})

    @SupportedDevices(["Ascend950"])
    def test_a5_supported_ops_without_blacklist(self):
        self.run_child_case("supported")

    @SupportedDevices(["Ascend950"])
    def test_a5_supported_ops_with_blacklist(self):
        self.run_child_case("supported", {"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"})

    @SupportedDevices(["Ascend950"])
    def test_a5_unsupported_ops_with_blacklist(self):
        self.run_child_case("unsupported", {"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"})

    @SupportedDevices(["Ascend950"])
    def test_a5_dropout_without_blacklist(self):
        self.run_child_case("dropout")

    @SupportedDevices(["Ascend950"])
    def test_a5_dropout_with_blacklist(self):
        self.run_child_case("dropout", {"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"})

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_force_incompatible_set_option_before_first_query(self):
        self.run_child_case("set_option_before")

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_force_incompatible_set_option_after_first_query(self):
        self.run_child_case("set_option_after")

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_global_compatible_switch_does_not_gate_randomness_disable(self):
        for value in ("0", "1"):
            self.run_child_case(
                "global_switch_ignored",
                {
                    "GLOBAL_COMPATIBLE": value,
                    "TORCH_NPU_FORCE_INCOMPATIBLE": "randomness",
                    "TORCH_NPU_USE_COMPATIBLE_IMPL": value,
                },
            )

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_force_incompatible_parser_ignores_empty_and_unknown_tokens(self):
        self.run_child_case("supported", {"TORCH_NPU_FORCE_INCOMPATIBLE": " randomness,,foo "})


if __name__ == "__main__":
    run_tests()
