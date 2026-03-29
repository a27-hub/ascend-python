torch_npu.npu_add_quant_gmm_（仅910D）
开发： 冯兴盼 30063590，测试：潘赛芸
商发版本号：Torino
使用场景：训练
产品支持情况
产品	是否支持
PR/Ascend 950DT	√

功能说明
	接口功能：在micro-batch训练场景，需要做micro-batch的梯度累计，会存在大量GroupedMatmul操作接InplaceAdd操作的融合场景。本算子（QuantGroupedMatmulInplaceAdd）将上述算子融合起来，以提高网络性能。
	计算公式：

关于量化技术的介绍参见4.3 量化介绍（刘波）。
−	T-C量化场景

−	mx量化场景

其中gsk代表K轴的mx量化的block size即32，Ki为每个分组的K的大小，x1_slicei代表x1i第m行长度为gsk的向量，x2_slicei代表x2i第n列长度为gsk的向量，K轴均从j*gsk起始切片，j的取值范围[0, k_loops)，k_loops=ceil(Ki/gsk)，支持最后的切片长度不足gsk。
函数原型
torch_npu.npu_add_quant_gmm_(self, x1, x2, x2_scale, group_list, *, x1_scale=None, group_list_type=0, group_sizes=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None) -> torch.Tensor
参数说明

参数说明里Shape使用的变量说明：
	g：表示分组数目，取值范围为1-1024。
	M：输出矩阵的倒数第二维大小，取值范围为1-2147483647。
	N：输出矩阵的倒数第一维大小，取值范围为1-2147483647。
	K：矩阵乘法reduce轴的大小，取值范围为1-2147483647。
注：M、N、K如为输入矩阵最内轴时需小于2097152。
参数（参数类型）：可选/必选参数，参数含义，dtype、shape、format、非连续、取值约束、其他约束等
	self（Tensor）：必选输入，待累加矩阵，数据类型支持float32，tensor支持3维，shape为(g, M, N)。数据格式支持ND。
	x1（Tensor）：必选输入，表示矩阵乘法中的左矩阵，数据类型支持float8_e5m2、float8_e4m3fn、hifloat8，tensor支持2维，shape为(K, M)，其中hifloat8需配置可选参数x1_dtype为对应类型，此时x1本身的dtype不再生效，但仍需保证x1本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
	x2（Tensor）：必选输入，表示矩阵乘法中的右矩阵，数据类型支持float8_e5m2、float8_e4m3fn、hifloat8，tensor支持2维，shape为(K, N)，其中hifloat8需配置可选参数x2_dtype为对应类型，此时x2本身的dtype不再生效，但仍需保证x2本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
	x2_scale（Tensor）：必选输入，表示矩阵乘法中的右矩阵的缩放因子，数据类型支持float8_e8m0fnu、float32，shape支持2维或3维，其中float8_e8m0fnu需配置可选参数x2_scale_dtype为对应类型，此时x2_scale本身的dtype不再生效，但仍需保证x2_scale本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
	group_list（Tensor）：必选输入，代表输入和输出分组轴方向的matmul大小分布, 数据类型支持int64, 1维tensor，shape为(g，), 数据格式支持ND。当group_list_type为0时，group_list必须为非负单调非递减数列；当group_list_type为1时，group_list必须为非负数列。
	x1_scale（Tensor）：可选输入，表示矩阵乘法中的左矩阵的缩放因子，数据类型支持float8_e8m0fnu、float32，shape支持1维、2维或3维，其中float8_e8m0fnu需配置可选参数x1_scale_dtype为对应类型，此时x1_scale本身的dtype不再生效，但仍需保证x1_scale本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
	group_list_type（int）：可选输入，代表group_list的表达形式。数据类型支持int32。
0：默认值，group_list中数值为分组轴大小的cumsum结果（累积和）。
1：group_list中数值为分组轴上每组大小。
	group_sizes （List[int]）：可选输入，列表大小为3，分别表示M、N、K三个轴上多少个参数共用一个缩放因子。数据类型支持int32，目前只支持默认值None。
	x1_dtype（int）：可选输入，可用于在x1无法用torch原生数据类型表示时显式指定x1的数据类型。
None：默认值，表示输入x1真实的数据类型与输入x1的dtype相同。
赋予输入x1真实的数据类型，当前仅支持hifloat8。
	x2_dtype（int）：可选输入，可用于在x2无法用torch原生数据类型表示时显式指定x2的数据类型。
None：默认值，表示输入x2真实的数据类型与输入x2的dtype相同。
赋予输入x2真实的数据类型，当前仅支持hifloat8。
	x1_scale_dtype（int）：可选输入，用于在x1_scale无法用torch原生数据类型表示时显式指定x1_scale的数据类型。
None：默认值，表示输入x1_scale真实的数据类型与输入x1_scale的dtype相同。
赋予输入x1_scale真实的数据类型，当前仅支持float8_e8m0fnu。
	x2_scale_dtype（int）：可选输入，用于在x2_scale无法用torch原生数据类型表示时显示指定x2_scale的数据类型。
None：默认值，表示输入x2_scale真实的数据类型与输入x2_scale的dtype相同。
赋予输入x2_scale真实的数据类型，当前仅支持float8_e8m0fnu。
返回值说明
self（Tensor）：groupedMatmul计算完成后与待累加矩阵相加得到的最后结果矩阵，支持数据类型、shape、数据格式均与输入self保持一致。
约束说明
	该接口支持训练场景下使用。
	该接口支持图模式（目前仅支持Pytorch2.1版本）。
	group_list第1维最大支持1024，即最多支持1024个group。
	数据类型约束：
场景	x1	x2	x2_scale	x1_scale	self
T-C量化	hifloat8	hifloat8	float32	float32	float32
mx量化	float8_e4m3fn/float8_e5m2	float8_e4m3fn/float8_e5m2	float8_e8m0fnu	float8_e8m0fnu	float32

	shape约束
场景	x2_scale	x1_scale
T-C量化	2维tensor，shape为(g, N)	2维tensor或1维tensor，shape为(g, 1)或(g,)
mx量化	3维tensor，shape为((K / 64) + g, N, 2)	3维tensor，shape为((K / 64) + g, M, 2)

调用示例
	单算子模式调用
−	T-C量化场景
import torch 
import torch_npu 
M = 576 
K = 512 
N = 7168 
g = 4 
y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu() 
x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).npu().transpose(0,1) 
x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).npu() 
x2_scale = torch.randint(-1, 1, (g, N), dtype=torch.float32).npu() 
x1_scale = torch.randint(-1, 1, (g,), dtype=torch.float32).npu() 
group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu() 
y = torch_npu.npu_add_quant_gmm_(y, x1, x2, x2_scale, group_list, x1_scale = x1_scale, group_list_type=0, x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8) 
print(y.cpu())
−	mx量化
import torch 
import torch_npu 
M = 576 
K = 512 
N = 7168 
g = 4 
y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu() 
x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(0,1) 
x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu() 
x2_scale = torch.randint(-1, 1, (int(K/64 + g), N, 2), dtype=torch.int8).npu() 
x1_scale = torch.randint(-1, 1, (int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0,1) 
group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu() 
y = torch_npu.npu_add_quant_gmm_(y, x1, x2, x2_scale, group_list, x1_scale = x1_scale, group_list_type=0, x1_scale_dtype=torch_npu.float8_e8m0fnu, x2_scale_dtype=torch_npu.float8_e8m0fnu) 
print(y.cpu())
	图模式调用
−	T-C量化场景
import torch 
import torch.nn as nn 
import torch_npu 
import torchair as tng 
from torchair.configs.compiler_config import CompilerConfig 
import os 

config = CompilerConfig() 
npu_backend = tng.get_npu_backend(compiler_config=config) 
#os.environ["ENABLE_ACLNN"] = "true" 
M = 576 
K = 512 
N = 7168 
g = 4 
class Model(nn.Module): 
    def __init__(self): 
        super().__init__() 
    def forward(self, y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, x1_dtype, x2_dtype): 
        return torch_npu.npu_add_quant_gmm_(y, x1.transpose(0,1), x2, x2_scale, group_list, x1_scale = x1_scale, group_list_type=group_list_type, x1_dtype=x1_dtype, x2_dtype=x2_dtype) 

def main(): 
    y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu() 
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).npu() 
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).npu() 
    x2_scale = torch.randint(-1, 1, (g, N), dtype=torch.float32).npu() 
    x1_scale = torch.randint(-1, 1, (g,), dtype=torch.float32).npu() 
    group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu() 
    group_list_type = 0 
    model = Model().npu() 
    model = torch.compile(model, backend=npu_backend) 
    y = model(y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, torch_npu.hifloat8, torch_npu.hifloat8) 
    print(y.cpu()) 

if __name__ == '__main__': 
    main()
−	mx量化
import torch 
import torch.nn as nn 
import torch_npu 
import torchair as tng 
from torchair.configs.compiler_config import CompilerConfig 
import os 

config = CompilerConfig() 
npu_backend = tng.get_npu_backend(compiler_config=config) 
#os.environ["ENABLE_ACLNN"] = "true" 
M = 576 
K = 512 
N = 7168 
g = 4 
class Model(nn.Module): 
    def __init__(self): 
        super().__init__() 
    def forward(self, y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, x1_scale_dtype, x2_scale_dtype): 
        return torch_npu.npu_add_quant_gmm_(y, x1.transpose(0,1), x2, x2_scale, group_list, x1_scale = x1_scale.transpose(0, 1), group_list_type=group_list_type, x1_scale_dtype=x1_scale_dtype, x2_scale_dtype=x2_scale_dtype) 

def main(): 
    y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu() 
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu() 
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu() 
    x2_scale = torch.randint(-1, 1, (int(K/64 + g), N, 2), dtype=torch.int8).npu() 
    x1_scale = torch.randint(-1, 1, (int(K/64 + g), M, 2), dtype=torch.int8).npu() 
    group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu() 
    group_list_type = 0 
    model = Model().npu() 
    model = torch.compile(model, backend=npu_backend) 
    y = model(y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, torch_npu.float8_e8m0fnu, torch_npu.float8_e8m0fnu) 
    print(y.cpu()) 

if __name__ == '__main__': 
    main()





ERROR: test_npu_fused_infer_attention_score_v2_A5_float8_e4m3fn_bsh_layout (__main__.TestFusedInferAttentionV2)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch_npu/testing/common_utils.py", line 206, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/home/c00879293/op-plugin/test/test_custom_ops/test_npu_fused_infer_attention_score_v2.py", line 530, in test_npu_fused_infer_attention_score_v2_A5_float8_e4m3fn_bsh_layout
    custom_output = torch_npu.npu_fused_infer_attention_score_v2(
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch/_ops.py", line 1158, in __call__
    return self._op(*args, **(kwargs or {}))
RuntimeError: npu_fused_infer_attention_score_v2_symint:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:464 NPU function error: call aclnnFusedInferAttentionScoreV5 failed, error code is 561002
[ERROR] 2047-12-19-03:01:45 (PID:2588035, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
EZ9999: Inner Error!
EZ9999[PID: 2588035] 2047-12-19-03:01:45.016.400 (EZ9999):  The shape of antiquant parameter[128] is not expected. Expect[1] when per-tensor mode.[FUNC:CheckKVAntiQuantParaShapeLegal][FILE:incre_flash_attention_tiling_v2.cpp][LINE:2391]
        TraceBack (most recent call last):
       Illegal shape of antiquant scale.[FUNC:CheckAntiQuantParam][FILE:incre_flash_attention_tiling_v2.cpp][LINE:2547]
       failed in IFA RunBigKernelTiling[FUNC:DoSubOpTiling][FILE:incre_flash_attention_tiling_v2.cpp][LINE:4452]
       failed in FIA DoSubOpTiling[FUNC:DoOpTiling][FILE:fused_infer_attention_score_tiling_v2.cpp][LINE:1021]
       FusedInferAttentionScore do tiling failed, ret is -1.
       Check NnopbaseExecutorDoTiling(executor) failed
       Check NnopbaseExecutorTilingAndUpdateBinInfo(executor) failed
       Check NnopbaseExecutorMatchCache(executor) failed
       Check NnopbaseRunForWorkspace(*executor, workspaceSize) failed


======================================================================
ERROR: test_npu_fused_infer_attention_score_v2_A5_float8_e5m2_antiquant (__main__.TestFusedInferAttentionV2)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch_npu/testing/common_utils.py", line 206, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/home/c00879293/op-plugin/test/test_custom_ops/test_npu_fused_infer_attention_score_v2.py", line 463, in test_npu_fused_infer_attention_score_v2_A5_float8_e5m2_antiquant
    custom_output = torch_npu.npu_fused_infer_attention_score_v2(
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch/_ops.py", line 1158, in __call__
    return self._op(*args, **(kwargs or {}))
RuntimeError: npu_fused_infer_attention_score_v2_symint:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:464 NPU function error: call aclnnFusedInferAttentionScoreV5 failed, error code is 161002
[ERROR] 2047-12-19-03:01:45 (PID:2588035, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 2588035] 2047-12-19-03:01:45.021.618 AclNN_Parameter_Error(EZ1001): Io input dtype or format is not supported, get io input info is query(DT_FLOAT16, ND) key(DT_FLOAT8_E5M2, ND) value(DT_FLOAT8_E5M2, ND) pseShiftOptional(NULL, NULL) attenMaskOptional(NULL, NULL) actualSeqLengthsOptional(NULL, NULL) actualSeqLengthsKvOptional(NULL, NULL) dequantScale1Optional(NULL, NULL) quantScale1Optional(NULL, NULL) dequantScale2Optional(NULL, NULL) quantScale2Optional(NULL, NULL) quantOffset2Optional(NULL, NULL) antiquantScaleOptional(NULL, NULL) antiquantOffsetOptional(NULL, NULL) blockTableOptional(NULL, NULL) queryPaddingSizeOptional(NULL, NULL) kvPaddingSizeOptional(NULL, NULL) keyAntiquantScaleOptional(DT_FLOAT16, ND) keyAntiquantOffsetOptional(NULL, NULL) valueAntiquantScaleOptional(DT_FLOAT16, ND) valueAntiquantOffsetOptional(NULL, NULL) keySharedPrefixOptional(NULL, NULL) valueSharedPrefixOptional(NULL, NULL) actualSharedPrefixLenOptional(NULL, NULL) queryRopeOptional(NULL, NULL) keyRopeOptional(NULL, NULL) keyRopeAntiquantScaleOptional(NULL, NULL) dequantScaleQueryOptional(NULL, 
        TraceBack (most recent call last):
