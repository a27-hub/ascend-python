import json
import os
import subprocess
import sys
import textwrap

from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


CHILD_SCRIPT = r"""
import hashlib
import json
import os
import sys

import torch
import torch_npu


SEED = 2026
SHAPE = (100, 100)


def tensor_md5(tensor):
    return hashlib.md5(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def fingerprint_supported_ops():
    result = {}

    torch.manual_seed(SEED)
    tensor = torch.empty(SHAPE, device="npu", dtype=torch.float32)
    tensor.normal_(0.0, 1.0)
    result["normal_"] = tensor_md5(tensor)

    torch.manual_seed(SEED)
    tensor = torch.randn(SHAPE, device="npu", dtype=torch.float32)
    result["randn"] = tensor_md5(tensor)

    torch.manual_seed(SEED)
    tensor = torch.empty(SHAPE, device="npu", dtype=torch.float32)
    torch.nn.init.xavier_uniform_(tensor)
    result["xavier_uniform_"] = tensor_md5(tensor)

    torch.manual_seed(SEED)
    tensor = torch.empty(SHAPE, device="npu", dtype=torch.float32)
    tensor.uniform_(-2.0, 3.0)
    result["uniform_"] = tensor_md5(tensor)

    torch.manual_seed(SEED)
    tensor = torch.empty(SHAPE, device="npu", dtype=torch.int32)
    tensor.random_(0, 100)
    result["random_from_to"] = tensor_md5(tensor)

    return result


def run_unsupported_smoke():
    tensor = torch.empty(SHAPE, device="npu", dtype=torch.int32)
    tensor.random_()
    tensor.cpu()

    tensor = torch.empty(SHAPE, device="npu", dtype=torch.float32)
    tensor.exponential_()
    tensor.cpu()

    if not hasattr(torch_npu, "npu_sim_exponential_"):
        return

    tensor = torch.empty(SHAPE, device="npu", dtype=torch.float32)
    try:
        torch_npu.npu_sim_exponential_(tensor, lambd=1.0)
    except RuntimeError as err:
        message = str(err).lower()
        if "npu_sim_exponential" in message and "support" in message:
            return
        raise
    tensor.cpu()


def run_dropout_smoke():
    torch.manual_seed(SEED)
    tensor = torch.ones(SHAPE, device="npu", dtype=torch.float32)
    torch.nn.functional.dropout(tensor, p=0.5, training=True).cpu()


def set_force_incompatible():
    torch_npu._C._npu_setOption({"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"})


def set_global_compatible():
    torch_npu.npu.use_compatible_impl(os.environ["GLOBAL_COMPATIBLE"] == "1")


def print_json(value):
    print(json.dumps(value, sort_keys=True))


def main():
    case = sys.argv[1]
    if case == "fingerprint":
        print_json(fingerprint_supported_ops())
    elif case == "set_option_before":
        set_force_incompatible()
        print_json(fingerprint_supported_ops())
    elif case == "set_option_after":
        before = fingerprint_supported_ops()
        set_force_incompatible()
        after = fingerprint_supported_ops()
        print_json({"before": before, "after": after})
    elif case == "global_switch":
        set_global_compatible()
        print_json(fingerprint_supported_ops())
    elif case == "unsupported":
        run_unsupported_smoke()
    elif case == "dropout":
        run_dropout_smoke()
    else:
        raise AssertionError("unknown child case: {}".format(case))


if __name__ == "__main__":
    main()
"""


class TestRandomnessCompatible(TestCase):
    def run_child(self, case, extra_env=None):
        env = os.environ.copy()
        env.pop("TORCH_NPU_FORCE_INCOMPATIBLE", None)
        env.pop("TORCH_NPU_USE_COMPATIBLE_IMPL", None)
        if extra_env is not None:
            env.update(extra_env)

        result = subprocess.run(
            [sys.executable, "-c", CHILD_SCRIPT, case],
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
        return result

    def run_child_json(self, case, extra_env=None):
        result = self.run_child(case, extra_env)
        stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
        for line in reversed(stdout_lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        self.fail("child case {} did not output json\nstdout:\n{}\nstderr:\n{}".format(
            case, result.stdout, result.stderr))

    @SupportedDevices(["Ascend910B"])
    def test_a2_randomness_blacklist_keeps_same_fingerprint(self):
        baseline = self.run_child_json("fingerprint")
        blacklist = self.run_child_json("fingerprint", {"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"})
        self.assertEqual(baseline, blacklist)

    @SupportedDevices(["Ascend950"])
    def test_a5_randomness_blacklist_changes_supported_fingerprint(self):
        baseline = self.run_child_json("fingerprint")
        blacklist = self.run_child_json("fingerprint", {"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"})
        self.assertEqual(set(baseline.keys()), set(blacklist.keys()))
        for op_name in baseline:
            self.assertNotEqual(
                baseline[op_name],
                blacklist[op_name],
                "{} output md5 should change after enabling randomness blacklist".format(op_name),
            )

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_force_incompatible_set_option_before_first_query(self):
        env_blacklist = self.run_child_json("fingerprint", {"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"})
        option_blacklist = self.run_child_json("set_option_before")
        self.assertEqual(env_blacklist, option_blacklist)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_force_incompatible_set_option_after_first_query_is_ignored(self):
        result = self.run_child_json("set_option_after")
        self.assertEqual(result["before"], result["after"])

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_global_compatible_switch_does_not_gate_randomness_blacklist(self):
        fingerprints = []
        for value in ("0", "1"):
            fingerprints.append(self.run_child_json(
                "global_switch",
                {
                    "GLOBAL_COMPATIBLE": value,
                    "TORCH_NPU_FORCE_INCOMPATIBLE": "randomness",
                    "TORCH_NPU_USE_COMPATIBLE_IMPL": value,
                },
            ))
        self.assertEqual(fingerprints[0], fingerprints[1])

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_force_incompatible_parser_ignores_empty_and_unknown_tokens(self):
        expected = self.run_child_json("fingerprint", {"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"})
        actual = self.run_child_json("fingerprint", {"TORCH_NPU_FORCE_INCOMPATIBLE": " randomness,,foo "})
        self.assertEqual(expected, actual)

    @SupportedDevices(["Ascend950"])
    def test_a5_unsupported_and_dropout_ignore_randomness_blacklist(self):
        env = {"TORCH_NPU_FORCE_INCOMPATIBLE": "randomness"}
        self.run_child("unsupported", env)
        self.run_child("dropout", env)


if __name__ == "__main__":
    run_tests()
