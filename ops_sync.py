import os
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).absolute().parent
OPS_DIR = BASE_DIR.joinpath("op_plugin/ops")

# from 2.1 to 2.4
OPS_VERSION_LIST = ["v2r1", "v2r2", "v2r3", "v2r4"]
DEPRECATION_OPS_DICT = {
    "v2r4": ["_aminmax"]
}


def selective_copy(src, dst, deprecation_ops):
    if src.is_dir():
        os.makedirs(dst, exist_ok=True)
        for path in src.iterdir():
            selective_copy(path, dst.joinpath(path.name), deprecation_ops)
    if src.is_file():
        if not dst.exists():
            is_deprecated = False
            for op_name in deprecation_ops:
                if dst.name.lower().startswith(op_name):
                    # op is deprecated
                    is_deprecated = True
            if not is_deprecated:
                shutil.copy2(src, dst)


# ops in v2r2 is baseline at present
def sync(initial_minor_version=2):
    src_ops_version = f"v2r{initial_minor_version}"
    for dst_ops_version in OPS_VERSION_LIST[initial_minor_version:]:
        selective_copy(
            OPS_DIR.joinpath(src_ops_version),
            OPS_DIR.joinpath(dst_ops_version),
            DEPRECATION_OPS_DICT.get(dst_ops_version, []))
        src_ops_version = dst_ops_version


if __name__ == "__main__":
    sync()
