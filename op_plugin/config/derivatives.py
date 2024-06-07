from ruamel.yaml import YAML
from operator import itemgetter
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", )
    parser.add_argument('-o', '--output_dir', help='output directory')
    parser.add_argument('-s', '--source_yaml', type=str, default=None, help='source yaml')
    args = parser.parse_args()

    VERSION = args.version
    output_dir = args.output_dir
    source_yaml = args.source_yaml

    yaml = YAML()
    yaml.width = 10000

    with open(source_yaml, 'r') as f:
        oldyaml = yaml.load(f)

    newyaml = []
    for func in oldyaml:
        newdict = func
        if VERSION in newdict['version']:
            del newdict['version']
            newyaml.append(newdict)


    with open(f'{output_dir}/new_derivatives.yaml', 'w') as f:
        yaml.dump(data=newyaml, stream=f)

if __name__ == '__main__':
    main()
