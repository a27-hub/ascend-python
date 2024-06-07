from ruamel.yaml import YAML
from operator import itemgetter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", )
args = parser.parse_args()

VERSION = args.version

yaml = YAML()
yaml.width = 10000

with open('derivatives.yaml', 'r') as f:
    oldyaml = yaml.load(f)

newyaml = []
for func in oldyaml:
    newdict = func
    if VERSION in newdict['version']:
        if 'npu_multi_head_attention(Tensor query, Tensor key, Tensor value, Tensor query_weight, Tensor key_weight, Tensor value_weight, Tensor attn_mask, Tensor out_proj_weight, Tensor? query_bias, Tensor? key_bias, Tensor? value_bias, Tensor? out_proj_bias, Tensor? dropout_mask, int attn_head_num, int attn_dim_per_head, int src_len, int tgt_len, float dropout_prob, bool softmax_use_float) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)' in newdict['name']:
            print(newdict)

        del newdict['version']
        if 'npu_multi_head_attention(Tensor query, Tensor key, Tensor value, Tensor query_weight, Tensor key_weight, Tensor value_weight, Tensor attn_mask, Tensor out_proj_weight, Tensor? query_bias, Tensor? key_bias, Tensor? value_bias, Tensor? out_proj_bias, Tensor? dropout_mask, int attn_head_num, int attn_dim_per_head, int src_len, int tgt_len, float dropout_prob, bool softmax_use_float) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)' in newdict['name']:
            print(newdict)
        newyaml.append(newdict)


with open('new_derivatives.yaml', 'w') as f:
    yaml.dump(data=newyaml, stream=f)
