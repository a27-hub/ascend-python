import os
import subprocess
import sys

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", "test_base_ops")

TEST_CASES = [
    ("test_lerp.py", "TestLerp.test_lerp_weight_0d_cpu_tensor"),
    ("test_lerp.py", "TestLerp.test_lerp_inplace_weight_0d_cpu_tensor"),
    ("test_logical_and.py", "TestLogicalAnd.test_logical_and_second_arg_0d_cpu_tensor"),
    ("test_logical_and.py", "TestLogicalAnd.test_logical_and_inplace_second_arg_0d_cpu_tensor"),
    ("test_rsub.py", "TestRsub.test_rsub_second_arg_0d_cpu_tensor"),
    ("test_rsub.py", "TestRsub.test_rsub_first_arg_0d_cpu_tensor"),
    ("test_atan2.py", "TestAtan2.test_atan2_second_arg_0d_cpu_tensor"),
    ("test_atan2.py", "TestAtan2.test_atan2_first_arg_0d_cpu_tensor"),
    ("test_atan2.py", "TestAtan2.test_atan2_inplace_second_arg_0d_cpu_tensor"),
    ("test_arctan2.py", "TestArctan2.test_arctan2_second_arg_0d_cpu_tensor"),
    ("test_arctan2.py", "TestArctan2.test_arctan2_first_arg_0d_cpu_tensor"),
    ("test_arctan2.py", "TestArctan2.test_arctan2_inplace_second_arg_0d_cpu_tensor"),
    ("test_bernoulli.py", "TestBernoulli.test_bernoulli_inplace_p_0d_cpu_tensor"),
    ("test_greater.py", "TestGreater.test_greater_first_arg_0d_cpu_tensor"),
    ("test_greater.py", "TestGreater.test_greater_inplace_second_arg_0d_cpu_tensor"),
    ("test_gt.py", "TestGt.test_gt_first_arg_0d_cpu_tensor"),
    ("test_gt.py", "TestGt.test_gt_inplace_second_arg_0d_cpu_tensor"),
    ("test_le.py", "TestLe.test_le_first_arg_0d_cpu_tensor"),
    ("test_le.py", "TestLe.test_le_inplace_second_arg_0d_cpu_tensor"),
    ("test_less.py", "TestLess.test_less_first_arg_0d_cpu_tensor"),
    ("test_less.py", "TestLess.test_less_inplace_second_arg_0d_cpu_tensor"),
    ("test_less_equal.py", "TestLessEqual.test_less_equal_first_arg_0d_cpu_tensor"),
    ("test_less_equal.py", "TestLessEqual.test_less_equal_inplace_second_arg_0d_cpu_tensor"),
    ("test_lt.py", "TestLt.test_lt_first_arg_0d_cpu_tensor"),
    ("test_lt.py", "TestLt.test_lt_inplace_second_arg_0d_cpu_tensor"),
    ("test_max.py", "TestMax.test_max_first_arg_0d_cpu_tensor"),
    ("test_max.py", "TestMax.test_maximum_first_arg_0d_cpu_tensor"),
    ("test_ne.py", "TestNe.test_ne_first_arg_0d_cpu_tensor"),
    ("test_ne.py", "TestNe.test_ne_inplace_second_arg_0d_cpu_tensor"),
    ("test_not_equal.py", "TestNotEqual.test_not_equal_first_arg_0d_cpu_tensor"),
    ("test_not_equal.py", "TestNotEqual.test_not_equal_inplace_second_arg_0d_cpu_tensor"),
    ("test_remainder.py", "TestRemainder.test_remainder_second_arg_0d_cpu_tensor"),
    ("test_remainder.py", "TestRemainder.test_remainder_first_arg_0d_cpu_tensor"),
    ("test_remainder.py", "TestRemainder.test_remainder_inplace_second_arg_0d_cpu_tensor"),
]

results = []

for file_name, test_case in TEST_CASES:
    test_path = os.path.join(TEST_DIR, file_name)
    class_name, method_name = test_case.split(".")
    node_id = f"{test_path}::{class_name}::{method_name}"
    cmd = [sys.executable, "-m", "pytest", node_id, "-v", "--tb=short"]
    label = f"{file_name}::{test_case}"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            results.append((label, "PASS"))
            print(f"[PASS] {label}")
        else:
            results.append((label, "FAIL"))
            print(f"[FAIL] {label}")
    except subprocess.TimeoutExpired:
        results.append((label, "TIMEOUT"))
        print(f"[TIMEOUT] {label}")
    except Exception as e:
        results.append((label, f"ERROR: {e}"))
        print(f"[ERROR] {label}: {e}")

print(f"\nPASS: {sum(1 for _, s in results if s == 'PASS')}/{len(results)}")
if fail_cases := [(l, s) for l, s in results if s != "PASS"]:
    print(f"FAIL: {len(fail_cases)}/{len(results)}")
    for label, status in fail_cases:
        print(f"  [{status}] {label}")
