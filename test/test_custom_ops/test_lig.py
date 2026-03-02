# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import torch
import torch_npu
import numpy as np
import torch.nn as nn
from precision_compare import data_compare
import sys
import datetime
import pandas as pd
from einops import rearrange
import openpyxl

DEVICE_ID = 6
torch_npu.npu.set_device(int(DEVICE_ID))

def log(text, file_name):
    print(text)
    log_file = open(file_name, 'a')
    # log_file.write(str(datetime.datetime.now()) + '  ')
    log_file.write(text)
    log_file.write("\n")
    log_file.close()

def print_log(data=None, file_name='auto_run.txt', level="INFO"):
    """
    print log
    :param data: value to print
    :param level: log level, include DEBUG, INFO(default), WARN and ERROR
    :return:
    """
    # level = "INFO"
    log_info = ("%s - [%s]-%s:%s - %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], level,
                                          os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                          str(sys._getframe().f_back.f_lineno).zfill(4), data))
    log(log_info, file_name)

def checkResult(a, b, name, num_print=16):
    print_log(f"info：开始计算 {name} 精度。")
    ratio_threshold = 0.005
    threshold_diff = 0.005
    rel_error_threshold = 0.2  # 相对误差20%
    print(f"a is {a.shape}")
    print(f"b is {b.shape}")
    success = False
    if a.shape == b.shape:
        a_flat = a.flatten().detach().cpu().numpy()
        b_flat = b.flatten().detach().cpu().numpy()
        diff_flat = np.abs(a_flat - b_flat)
        rel_error = diff_flat / np.maximum(np.abs(b_flat), 1e-8)

        # 找出相对误差超过20%的索引
        idx_large_error = np.where(rel_error > rel_error_threshold)[0]
        n = min(num_print, len(idx_large_error))
        totalCount = min(num_print, len(a_flat))
        print_log(f"debug: {name} 相对误差超过 {rel_error_threshold*100:.1f}% 的前 {n} 个元素：")
        print_log("        idx |      npu值       |      cpu值       |    abs_diff    |   rel_error(%)")
        print_log("       -----+------------------+------------------+----------------+----------------")
        diff_val = abs(a_flat[0] - b_flat[0]) 
        for i in range(totalCount): # TEST
            print_log(f"       {i:4d} | {a_flat[i]:14.8f} | {b_flat[i]:14.8f} | {diff_flat[i]:12.8f} | {rel_error[i]*100:12.4f}")
        # for i in idx_large_error[:n]:
        #     print_log(f"       {i:4d} | {a_flat[i]:14.8f} | {b_flat[i]:14.8f} | {diff_flat[i]:12.8f} | {rel_error[i]*100:12.4f}")

        # 精度统计逻辑保持不变
        if torch.all(torch.eq(a, b)):
            print_log(f"info：{name} 计算结果与标杆完全相同。")
            ratio_diff = 0
            diff = 0
            return 1, 0, 0, True
        else:
            diff = torch.abs(a.sub(b))
            min_tensor = torch.empty(a.shape, dtype=torch.float16).to(a.device)
            min_tensor.fill_(0.000025)
            max_val = torch.max(torch.abs(a), torch.abs(b))
            threshold = torch.max(max_val.mul(ratio_threshold), min_tensor)
            mask = diff > threshold
            num_diff = torch.sum(mask)
            ratio_diff = num_diff / torch.numel(a)

            if ratio_diff > threshold_diff:
                print_log(f"error：{name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%}，超过阈值，用例执行失败！！！")
            else:
                print_log(f"info：{name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%}。")
                success = True

        ratio = (1 - ratio_diff).cpu().detach().numpy()
        max_diff = torch.max(diff).cpu().detach().numpy()
        print_log(f"{name} diff_max : {max_diff:.8f}")
        sum_diff = torch.sum(diff).cpu().detach().numpy()
        print_log(f"{name} diff_sum : {sum_diff:.6f}")
    else:
        print_log(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print_log(f"debug: a {a.shape}")
        print_log(f"debug: b {b.shape}")
        ratio = 0
        max_diff = 999999
        sum_diff = 999999

    return ratio, max_diff, sum_diff, success


def get_col_index(table, col_name):
    col_index = None
    # print table
    for i in range(1, table.max_column + 1):
        if table.cell(row=1, column=i).value == col_name:
            col_index = i
            break
    return col_index

class LightningIndexerGrad():

    def __init__(self, batch, seqlenQ, seqlenK, headNumQ, headNumK, groupNum, headDim, indexTopk, inputDataType, result_xlsx_name, case_name, device):
        self.batch = batch
        self.seqlenQ = seqlenQ
        self.seqlenK = seqlenK
        self.headNumQ = headNumQ
        self.headNumK = headNumK
        self.groupNum = groupNum
        self.headDim = headDim
        self.indexTopk = indexTopk
        self.inputDataType = inputDataType
        self.result_xlsx_name = result_xlsx_name
        self.case_name = case_name
        self.device = device

    def lightning_indexer_fwd(self, q, k, weights, indexTopk, layout):
        weights = weights.reshape(*weights.shape, 1)
        mask = torch.full((self.seqlenQ, self.seqlenK), float("-inf")).triu_(1) if self.seqlenQ > 1 else None
        query = q.permute(0, 2, 1, 3) # (B, S1, N1, D) -> (B, N1, S1, D)
        key = k.permute(0, 2, 3, 1) # (B, S2, N2, D) -> (B, N2, D, S2)
        p = torch.matmul(query, key) # (B, N1, S1, S2)
        reluOut = torch.nn.functional.relu(p) # (B, N1, S1, S2) 
        weightOut = reluOut * weights.permute(0, 2, 1, 3) # (B, N1, S1, S2) * (B, N1, S1, 1) -> (B, N1, S1, S2)
        reduceOut = torch.sum(weightOut, dim=1) # (B, S1, S2)
        indexScore = reduceOut
        if mask is not None:
            indexScore += mask # (B, S1, S2) + (S1, S2) -> (B, S1, S2)

        # N2 = 1
        value2, topkIndices2 = indexScore.topk(min(self.indexTopk, self.seqlenK), dim=-1) # (B, S1, S2) -> (B, S1, topK)
        return value2, topkIndices2

    def sum_pairwise_to_eight(self, tensor, dim=-1, target_size=8, compute_dtype=torch.float32):
        original_dtype = tensor.dtype
        original_device = tensor.device
        
        size = tensor.shape[dim]
        if size <= target_size:
            return torch.sum(tensor, dim=dim)
        
        t = tensor.to(dtype=compute_dtype)
        
        dim_normalized = dim if dim >= 0 else tensor.ndim + dim
        if dim_normalized != tensor.ndim - 1:
            t = t.transpose(dim_normalized, -1)
        
        current_size = t.shape[-1]
        while current_size > target_size:
            half_size = current_size // 2
            first_half = t[..., :half_size]
            second_half = t[..., half_size:half_size + half_size]
            
            first_half += second_half
            t = first_half
            current_size = half_size
        
        result = torch.sum(t, dim=-1)
        result = result.to(original_dtype)
        if dim_normalized != tensor.ndim - 1:
            result = result.transpose(dim_normalized, -1)
        return result

    def lightning_indexer_bwd(self, valueGrad, q, k, weights, topkIndices, indexTopk, layout, isHighPrecision=False, useMatmul=True):
        valueGrad = valueGrad.unsqueeze(1).expand(self.batch, self.headNumQ, self.seqlenQ, self.indexTopk)
        valueGrad = valueGrad.to(self.inputDataType)
        topkIndices = topkIndices.unsqueeze(1).expand(self.batch, self.headNumQ, self.seqlenQ, self.indexTopk)
        dumpOutput = None
        dumpWeightsOutput = None
        mulGrad = torch.zeros((self.batch, self.headNumQ, self.seqlenQ, self.indexTopk)).to(torch.float32)
        if useMatmul:
            print("ENTER MULGRAD CUBE")
            mulGrad = mulGrad.permute(0, 2, 1, 3).reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum, self.indexTopk)
            for bIdx in range(self.batch):
                for headNumKIdx in range(self.headNumK):
                    for seqlenIdx in range(self.seqlenQ):
                        weightSelect = weights.reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum)[bIdx, seqlenIdx, headNumKIdx, :].reshape(self.groupNum, 1)
                        valueGradSelect = valueGrad[bIdx, headNumKIdx, seqlenIdx, :].reshape(1, self.indexTopk)
                        mulGrad[bIdx, seqlenIdx, headNumKIdx, :, :] = torch.matmul(weightSelect.to(self.inputDataType), valueGradSelect.to(self.inputDataType))
            mulGrad = mulGrad.reshape(self.batch, self.seqlenQ, self.headNumQ, self.indexTopk).permute(0, 2, 1, 3)
        else:
            print("ENTER MULGRAD VECTOR")
            weights = weights.reshape(*weights.shape, 1)
            mulGrad = valueGrad * weights.permute(0, 2, 1, 3)

        reluIn = torch.zeros((self.batch, self.headNumK, self.seqlenQ, self.groupNum, self.indexTopk)).to(torch.float32).npu()
        for bIdx in range(self.batch):
            for headNumKIdx in range(self.headNumK):
                for seqlenIdx in range(self.seqlenQ):
                    indices = topkIndices[bIdx, headNumKIdx, seqlenIdx, :]
                    kSelect = torch.index_select(k.permute(0, 2, 1, 3)[bIdx, headNumKIdx, :], dim=0, index=indices).reshape(self.indexTopk, self.headDim)
                    qSelect = q.reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum, self.headDim)[bIdx, seqlenIdx, headNumKIdx, :, :].reshape(self.groupNum, self.headDim)
                    reluInPart = torch.matmul(qSelect.to(self.inputDataType).to(torch.float), kSelect.permute(1, 0).to(self.inputDataType).to(torch.float))
                    reluIn[bIdx, headNumKIdx, seqlenIdx, :, :] = reluInPart
        
        reluIn = reluIn.permute(0, 1, 3, 2, 4).reshape(self.batch, self.headNumQ, self.seqlenQ, self.indexTopk).npu()
        mask = reluIn < 0.0
        reluGrad = torch.masked_fill(mulGrad.npu(), mask.npu(), 0.0).npu()

        keyPermute = k.permute(0, 2, 1, 3)
        dq = torch.zeros((self.batch, self.headNumK, self.seqlenQ, self.groupNum, self.headDim)).to(torch.float32).npu()
        for bIdx in range(self.batch):
            for headNumKIdx in range(self.headNumK):
                for seqlenIdx in range(self.seqlenQ):
                    indices = topkIndices[bIdx, headNumKIdx, seqlenIdx, :]
                    selectKey = torch.index_select(keyPermute[bIdx, headNumKIdx, :, :], 0, indices)
                    permuteReluGrad = reluGrad.permute(0, 2, 1, 3).reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum, self.indexTopk)[bIdx, seqlenIdx, headNumKIdx, :, :]
                    # dqPart shape is [groupNum, headDim]
                    dqPart = torch.matmul(permuteReluGrad.reshape(self.groupNum, self.indexTopk).to(self.inputDataType).npu(), selectKey.reshape(self.indexTopk, self.headDim).to(self.inputDataType).npu())
                    dq[bIdx, headNumKIdx, seqlenIdx, :, :] = dqPart.reshape(self.groupNum, self.headDim)
        dq = dq.permute(0, 1, 3, 2, 4).reshape(self.batch, self.headNumQ, self.seqlenQ, self.headDim)

        # dk single block output
        dk = torch.zeros((self.batch, self.headNumK, self.seqlenK, self.headDim)).to(torch.float32).npu()
        for bIdx in range(self.batch):
            for headNumKIdx in range(self.headNumK):
                for seqlenIdx in range(self.seqlenQ):
                    indices = topkIndices[bIdx, headNumKIdx, seqlenIdx, :]
                    selectQuery = q.reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum, self.headDim)[bIdx, seqlenIdx, headNumKIdx, :, :]
                    permuteReluGrad = reluGrad.permute(0, 2, 1, 3).reshape(self.batch, self.seqlenQ, self.headNumK, self.groupNum, self.indexTopk)[bIdx, seqlenIdx, headNumKIdx, :, :].permute(1, 0)
                    # dqPart shape is [indexTopk, headDim]
                    dkPart = torch.matmul(permuteReluGrad.reshape(self.indexTopk, self.groupNum).to(self.inputDataType).to(torch.float).npu(), selectQuery.reshape(self.groupNum, self.headDim).to(self.inputDataType).to(torch.float).npu())
                    dk[bIdx, headNumKIdx].index_add_(0, indices.npu(), dkPart.float().npu())

        reluOut = torch.nn.functional.relu(reluIn)
        dweights = reluOut * valueGrad.to(torch.float32)
        dweights = self.sum_pairwise_to_eight(dweights.to(torch.float32)).reshape(self.batch, self.headNumQ, self.seqlenQ, 1)
        return dq.permute(0, 2, 1, 3), dk.permute(0, 2, 1, 3), dweights.permute(0, 2, 1, 3), dumpOutput, dumpWeightsOutput

    def test(self):
        q = torch.randn(self.batch, self.seqlenQ, self.headNumQ, self.headDim).to(torch.float32)
        k = torch.randn(self.batch, self.seqlenK, self.headNumK, self.headDim).to(torch.float32)
        weights = torch.randn(self.batch, self.seqlenQ, self.headNumQ).to(torch.float32)
        valueGrad = torch.randn((self.batch, self.seqlenQ, self.indexTopk)).to(torch.float32)
        layoutStr = "BSND"
        sparseMode = 3
        isHighPrecision = True

        loss, topkIndices = self.lightning_indexer_fwd(q.clone(), k.clone(), weights.clone(), self.indexTopk, "BSND")
        dq, dk, dweights, dumpOutput, dumpWeightsOutput = self.lightning_indexer_bwd(valueGrad.npu(), q.clone().npu(), k.clone().npu(), weights.clone().npu(), topkIndices.npu(), self.indexTopk, "BSND", isHighPrecision, True)
        dq = dq.detach().cpu()
        dk = dk.detach().cpu()
        dweights = dweights.detach().cpu()

        is_test_fusedop = True
        if is_test_fusedop:
            if layoutStr == "BSND":
                dq_npu, dk_npu, dweights_npu = torch_npu.npu_lightning_indexer_grad(
                    q.to(self.inputDataType).clone().npu(),
                    k.to(self.inputDataType).clone().npu(),
                    valueGrad.to(self.inputDataType).clone().npu(),
                    topkIndices.to(torch.int32).clone().npu(),
                    weights.to(self.inputDataType).clone().npu(),
                    actual_seq_lengths_query=None, 
                    actual_seq_lengths_key=None,
                    layout=layoutStr,
                    sparse_mode=sparseMode
                )
            else:
                actualSeqLengthQ = torch.tensor([seqlenQ * i for i in range(1, batch + 1)], dtype=torch.int32)
                actualSeqLengthK = torch.tensor([seqlenK * i for i in range(1, batch + 1)], dtype=torch.int32)
                dq_npu, dk_npu, dweights_npu = torch_npu.npu_lightning_indexer_grad(
                    q.reshape(-1, self.headNumQ, self.headDim).to(self.inputDataType).clone().npu(),
                    k.reshape(-1, self.headNumK, self.headDim).to(self.inputDataType).clone().npu(),
                    valueGrad.reshape(-1, self.indexTopk).to(self.inputDataType).clone().npu(),
                    topkIndices.reshape(-1, self.indexTopk).to(torch.int32).clone().npu(),
                    weights.reshape(-1, self.headNumQ).to(self.inputDataType).clone().npu(),
                    actual_seq_lengths_query=actualSeqLengthQ.clone().npu(), 
                    actual_seq_lengths_key=actualSeqLengthK.clone().npu(),
                    layout=layoutStr,
                    sparse_mode=sparseMode
                )

            precision = 'failed'
            dq_suceess = False
            dk_suceess = False
            dw_suceess = False
            # test all precision
            print("compare dq precision")
            data_compare(dq_npu.detach().cpu().float().numpy(), dq.detach().float().numpy())
            print("compare dk precision")
            data_compare(dk_npu.detach().cpu().float().numpy(), dk.detach().float().numpy())
            print("compare dweights precision")
            data_compare(dweights_npu.detach().cpu().float().numpy(), dweights.to(self.inputDataType).detach().float().numpy())

            rst0, max0, sum0, dq_suceess = checkResult(dq_npu.detach().cpu().float().reshape(-1), dq.to(self.inputDataType).detach().float().reshape(-1), "dquery")
            rst1, max1, sum1, dk_suceess = checkResult(dk_npu.detach().cpu().float().reshape(-1), dk.to(self.inputDataType).detach().float().reshape(-1), "dkey")
            rst2, max2, sum2, dw_suceess = checkResult(dweights_npu.detach().cpu().float().reshape(-1), dweights.to(self.inputDataType).detach().float().reshape(-1), "dweights")

            if dq_suceess and dk_suceess and dw_suceess:
                precision = 'pass'

            result_file = self.result_xlsx_name
            print("result_file is ", result_file)
            df = pd.read_excel(result_file, sheet_name='Sheet1')
            
            df.loc[df['Testcase_Name'] == case_name, 'precision'] = str(precision)

            df.loc[df['Testcase_Name'] == case_name, 'Actual_dquery_pricision'] = rst0
            df.loc[df['Testcase_Name'] == case_name, 'Actual_dkey_pricision'] = rst1
            df.loc[df['Testcase_Name'] == case_name, 'Actual_dweights_pricision'] = rst2

            df.loc[df['Testcase_Name'] == case_name, 'Actual_dquery_err_max'] = max0
            df.loc[df['Testcase_Name'] == case_name, 'Actual_dkey_err_max'] = max1
            df.loc[df['Testcase_Name'] == case_name, 'Actual_dweights_err_max'] = max2

            df.loc[df['Testcase_Name'] == case_name, 'Actual_dquery_err_sum'] = sum0
            df.loc[df['Testcase_Name'] == case_name, 'Actual_dkey_err_sum'] = sum1
            df.loc[df['Testcase_Name'] == case_name, 'Actual_dweights_err_sum'] = sum2

            df.to_excel(result_file, sheet_name='Sheet1', index=False)

if __name__ == "__main__":

    headNumQ = 64
    headNumK = 1
    groupNum = headNumQ // headNumK
    headDim = 128
    indexTopk = 2048
    device = "npu"
    inputDataType = torch.bfloat16
        
    case_path = f"./LightningIndexerGrad.xlsx"
    result_xlsx_name =f"./output_xls/LightningIndexerGrad_Result_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.xlsx"
    os.system(f"cp {case_path} {result_xlsx_name}")

    data = openpyxl.load_workbook(case_path)
    sheet = 'Sheet1'
    table = data[sheet]
    nrows = table.max_row
    Testcase_Name_Col = get_col_index(table, 'Testcase_Name')
    B_Col = get_col_index(table, 'B')
    S1_Col = get_col_index(table, 'S1')
    S2_Col = get_col_index(table, 'S2')
    IS_ENABLE = get_col_index(table, 'Enable')
    for i in range(2, nrows + 1):
        case_name = table.cell(i, Testcase_Name_Col).value
        bValue = int(table.cell(i, B_Col).value)
        s1Value = int(table.cell(i, S1_Col).value)
        s2Value = int(table.cell(i, S2_Col).value)

        isEnable = table.cell(i, IS_ENABLE).value
        if (isEnable == "enable"):
            op = LightningIndexerGrad(bValue, s1Value, s2Value, headNumQ, headNumK, groupNum, headDim, indexTopk, inputDataType, result_xlsx_name, case_name, device)
            op.test()
