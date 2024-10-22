from operator import itemgetter
import argparse
import os
import stat
import yaml

from codegen.utils import PathManager


def get_version(ver, all_version):
    if isinstance(ver, list):
        start_ver = all_version.index(ver[0])
        end_ver = None if ver[1] == 'newest' else all_version.index(ver[1])
        real_ver = all_version[start_ver:end_ver]
    elif ver == 'all_version':
        real_ver = all_version
    else:
        real_ver = ver.split(', ')

    return real_ver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", )
    parser.add_argument('-o', '--output_dir', help='output directory')
    parser.add_argument('-s', '--source_yaml', type=str, default=None, help='source yaml')
    args = parser.parse_args()
    pytorch_version = args.version.split('.')
    version = f"v{pytorch_version[0]}.{pytorch_version[1]}"
    output_dir = args.output_dir
    source_yaml = args.source_yaml

    source_yaml_path = os.path.realpath(source_yaml)
    PathManager.check_directory_path_readable(source_yaml_path)
    with open(source_yaml_path, 'r') as f:
        old_yaml = yaml.safe_load(f)

    all_version = old_yaml['all_version']
    new_yaml = []
    for item in old_yaml['backward']:
        new_item = item
        if version in get_version(new_item['version'], all_version):
            del new_item['version']
            new_yaml.append(new_item)

    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(f'{output_dir}/derivatives.yaml', flags, modes), 'w') as f:
        yaml.dump(data=new_yaml, stream=f, width=2000, sort_keys=False)


if __name__ == '__main__':
    main()
