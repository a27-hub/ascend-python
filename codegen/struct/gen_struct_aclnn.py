
import argparse
import yaml

from codegen.gen import parse_native_yaml, FileManager, parse_native_yaml_struct, get_grouped_native_functions
from codegen.utils import concatMap
from codegen.struct.struct_codegen import parse_struct_yaml, gen_op_api


def main() -> None:

    parser = argparse.ArgumentParser(description='Generate struct aclnn files')
    parser.add_argument(
        '-s',
        '--native_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '--struct_yaml',
        help='path to struct yaml file containing aclnn operators struct definitions')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    options = parser.parse_args()

    fm = FileManager(
            install_dir=options.output_dir, template_dir="codegen/struct/templates", dry_run=False
        )

    with open(options.native_yaml, "r") as f:
        es = yaml.safe_load(f)
    native_functions = parse_native_yaml_struct(es)
    grouped_native_functions = get_grouped_native_functions(native_functions)

    struct_info = parse_struct_yaml(options.struct_yaml, grouped_native_functions)
    gen_op_api(fm, struct_info)


if __name__ == '__main__':
    main()
