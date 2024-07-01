import re
import argparse
import os
import stat
from ruamel.yaml import YAML

OP_API = 'op_api'
ACL_OP = 'acl_op'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=str, default=None, help='pytorch version')
    parser.add_argument('-o', '--output_dir', help='output directory')
    parser.add_argument('-s', '--source_yaml', type=str, default=None, help='source yaml')
    args = parser.parse_args()
    pytorch_version = args.version.split('.')
    version = f"v{pytorch_version[0]}.{pytorch_version[1]}"
    output_dir = args.output_dir
    source_yaml = args.source_yaml

    yaml = YAML()
    yaml.width = 1000
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(source_yaml, 'r') as f:
        oldyaml = yaml.load(f)

    newyaml = {'official':[], 'custom':[], 'symint':[], 'tocpu':[], 'unsupported':[], 'quant':[]}

    string = ['official', 'custom', 'symint', 'quant']

    # official,custom,symint
    for key in string:
        for func in oldyaml[key]:
            if version in func['version']:
                newdict = func
                if func.get('impl_ns') and func.get('version'):
                    impl_nslist = re.split(', ', func['impl_ns'])
                    versionlist = re.split(', ', func['version'])
                    versionidx = [idx for idx,val in enumerate(versionlist) if val == version]
                    supportversion = [impl_nslist[i] for i in versionidx]
                    newdict['impl_ns'] = ', '.join(supportversion)
                del newdict['kernel']
                del newdict['version']
                if (version == 'v1.11' or version == 'v2.0') and func.get('sparse'):
                    del newdict['sparse']

                if func.get('exposed') and version in versionlist:
                    newdict['exposed'] = 'True'
                elif func.get('exposed'):
                    del newdict['sparse']

                newyaml.get(key).append(newdict)

    # tocpu and unsupported
    for key in ['tocpu', 'unsupported']:
        for func in oldyaml[key]:
            if version in func['version']:
                newyaml.get(key).append(func['func'])

    # save to yaml
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(f'{output_dir}/new_op_plugin_functions.yaml', flags, modes), 'w') as f:
        yaml.dump(data=newyaml, stream=f)

if __name__ == '__main__':
    main()
