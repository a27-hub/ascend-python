commAlg 支持""，"fullmesh_v1"，"fullmesh_v2"三种输入方式。""：默认值，不使能fullmesh_v2模板；"fullmesh_v1"：不使能fullmesh_v2模板；"fullmesh_v2"：使能fullmesh_v2模板，其中commAlg仅支持tpWorldSize为1场景。
xActiveMaskOptional 要求为1D或2D Tensor（1D时shape为(Bs, )，2D时shape为(Bs, K)）；1D时true需排在false前，2D时token对应K个值全为false则不参与通信。
expertScalesOptional 当前版本不支持，传空指针即可。
epWorldSize 取值范围[2, 768]。
moeExpertNum 取值范围(0, 1024]。
groupTp 当前版本不支持，传空字符即可。
tpWorldSize 当前版本不支持，传0即可。
tpRankId 当前版本不支持，传0即可。
expertShardType 当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
sharedExpertNum 当前取值范围[0, 4]。
sharedExpertRankNum 取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。
epRecvCountsOut 的shape为(epWorldSize * max(tpWorldSize, 1) * localExpertNum,)。
有TP域通信时tpRecvCountsOut为1D shape Tensor，shape为(tpWorldSize,)。
expandScalesOut 当前版本不支持该输出。
quantMode 支持0（非量化）、1（静态量化）、2（pertoken动态量化）、3（pergroup动态量化）、4（mxfp8动态量化）。
elasticInfoOptional 当前版本不支持，传空指针即可。
zeroExpertNum 取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是[moeExpertNum, moeExpertNum + zeroExpertNum)。
copyExpertNum 取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1，专家ID范围[moeExpertNum + zeroExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum)。
constExpertNum 取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1，专家ID范围[moeExpertNum + zeroExpertNum + copyExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)。
performanceInfoOptional 预留参数，当前版本不支持，传空指针即可。
scalesOptional 非量化、MX量化场景传空指针；静态量化必传有效数据；pertoken、pergroup动态量化场景可传有效数据或空指针