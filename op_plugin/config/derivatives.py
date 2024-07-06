from operator import itemgetter
import argparse
import os
import stat
import yaml


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


    with open(source_yaml, 'r') as f:
        oldyaml = yaml.safe_load(f)

    newyaml = []
    for func in oldyaml:
        newdict = func
        if version in newdict['version']:
            del newdict['version']
            newyaml.append(newdict)

    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(f'{output_dir}/new_derivatives.yaml', flags, modes), 'w') as f:
        yaml.dump(data=newyaml, stream=f)

if __name__ == '__main__':
    main()
