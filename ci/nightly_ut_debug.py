#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Nightly UT debug wrapper.
Runs environment diagnostics before executing access_control_test.py.
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent.parent


def run_check(label, cmd):
    print(f"\n=== {label} ===")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            print(f"[WARN] {label} exited with code {result.returncode}")
    except Exception as e:
        print(f"[WARN] {label} failed: {e}")


def main():
    print("=" * 60)
    print("Nightly UT Debug - Environment Check")
    print("=" * 60)

    # 1. NPU devices
    run_check("NPU-SMI", "npu-smi info")
    run_check("Davinci Devices", "ls -la /dev/davinci* 2>/dev/null || echo 'No davinci devices found'")

    # 2. Ascend CANN
    run_check("Ascend Directory", "ls /usr/local/Ascend/ 2>/dev/null || echo 'Ascend dir not found'")
    run_check("CANN Version", "cat /usr/local/Ascend/cann/version.info 2>/dev/null || echo 'CANN version.info not found'")

    # 3. torch_npu device check
    print("\n=== torch_npu Device Check ===")
    try:
        import torch_npu
        print(f"torch_npu version: {torch_npu.__version__}")
        device_count = torch_npu.npu.device_count()
        print(f"NPU device count: {device_count}")
        if device_count > 0:
            for i in range(device_count):
                print(f"  Device {i}: {torch_npu.npu.get_device_name(i)}")
        else:
            print("[WARN] No NPU devices detected by torch_npu")
    except Exception as e:
        print(f"[WARN] torch_npu device check failed: {e}")

    # 4. modify_files.txt
    print("\n=== modify_files.txt ===")
    modify_files_path = os.path.join(BASE_DIR, "modify_files.txt")
    if os.path.exists(modify_files_path):
        with open(modify_files_path) as f:
            lines = f.readlines()
        print(f"Total files: {len(lines)}")
        print(f"First 5 files: {[l.strip() for l in lines[:5]]}")
    else:
        print("[WARN] modify_files.txt not found!")

    print("\n" + "=" * 60)
    print("Environment check done. Starting UT...")
    print("=" * 60 + "\n")

    # Run original access_control_test.py with same args
    original_script = os.path.join(BASE_DIR, "ci", "access_control_test.py")
    sys.argv[0] = original_script
    os.chdir(BASE_DIR)
    exec(open(original_script).read())


if __name__ == "__main__":
    main()
