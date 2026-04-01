Process SpawnProcess-2:
Traceback (most recent call last):
  File "/root/miniconda3/envs/cstest/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/root/miniconda3/envs/cstest/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/c00879293/op-plugin/test/test_custom_ops/test_moe_distribute_dispatch_v2.py", line 56, in _test_npu_moe_distribute_dispatch_v2
    out, _, _, _, _, _, _ = torch_npu.npu_moe_distribute_dispatch_v2(x=x,
  File "/root/miniconda3/envs/cstest/lib/python3.10/site-packages/torch/_ops.py", line 1158, in __call__
    return self._op(*args, **(kwargs or {}))
RuntimeError: npu_moe_distribute_dispatch_v2:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:184 NPU function error: call aclnnMoeDistributeDispatchV4 failed, error code is 561002
[ERROR] 2026-04-01-17:17:22 (PID:3385337, Device:1, RankID:-1) ERR00100 PTA call acl api failed.
EZ9999: Inner Error!
EZ9999[PID: 3385337] 2026-04-01-17:17:22.284.307 (EZ9999):  expertIdShape's dim1(k) should be in (0, min(16, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum = 3)], but got expertIdShape's dim1=4.[FUNC:CheckTensorShape][FILE:moe_distribute_dispatch_v2_tiling.cpp][LINE:1132]
        TraceBack (most recent call last):
       Check tensor shape failed.[FUNC:MoeDistributeDispatchA3TilingFuncImplPublic][FILE:moe_distribute_dispatch_v2_tiling.cpp][LINE:1688]
       MoeDistributeDispatchV2 do tiling failed, ret is -1.
       Check NnopbaseExecutorDoTiling(executor) failed
       Check NnopbaseExecutorTilingAndUpdateBinInfo(executor) failed
       Check NnopbaseExecutorMatchCache(executor) failed
       Check NnopbaseRunForWorkspace(*executor, workspaceSize) failed
