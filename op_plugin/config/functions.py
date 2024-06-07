from ruamel.yaml import YAML
from operator import itemgetter
import re

yaml = YAML()
yaml.width = 1000

with open('op_plugin_functions.yaml', 'r') as f:
    # yaml111 = YAML()
    oldyaml = yaml.load(f)

VERSION = 'v2.1'
OP_API = 'op_api'
ACL_OP = 'acl_op'

newyaml = {'official':[], 'custom':[], 'symint':[], 'tocpu':[], 'unsupported':[]}

string = ['official', 'custom', 'symint',
    # 'tocpu', 'unsupported'
    ]

for key in string:
    for func in oldyaml[key]:
        if VERSION in func['version']:
            newdict = {}
            newdict['func'] = func['func']
            impl_nslist = re.split(', ', func['impl_ns'])
            versionlist = re.split(', ', func['version'])
            versionidx = [idx for idx,val in enumerate(versionlist) if val == VERSION]
            supportversion = [impl_nslist[i] for i in versionidx]
            if ACL_OP in supportversion and OP_API in supportversion:
                newdict['impl_ns'] = f'{ACL_OP}, {OP_API}'
            elif ACL_OP in supportversion:
                newdict['impl_ns'] = ACL_OP
            elif OP_API in supportversion:
                newdict['impl_ns'] = OP_API
            newyaml[key].append(newdict)

for func in oldyaml['tocpu']:
    if VERSION in func['version']:
        newyaml['tocpu'].append(func['func'])

for func in oldyaml['unsupported']:
    if VERSION in func['version']:
        newyaml['unsupported'].append(func['func'])

# print(newyaml)
with open('new_functions.yaml', 'w') as f:
    yaml.dump(data=newyaml, stream=f)

