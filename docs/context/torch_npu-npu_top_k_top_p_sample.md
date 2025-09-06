# torch_npu-npu_top_k_top_p_sample.md

## 功能说明

-   根据输入词频logits、top-k/top-p采样参数、随机采样权重分布q，进行topk-topp-sample采样计算，输出每个batch的最大词频logits_select_idx，以及topk-topp采样后的词频分布logits_top_kp_select。

-   输入输出分为三个可选的模块选项，以下按它们的在完整topk-topp-sample功能下的计算顺序，分别说明具体支持场景：

    - topK模块：
        - 外部输入：
        logits:[batch, voc_size]
        top_k:[batch, 1]
        
        - 中间结果（不输出）：
        logits_sort:[batch, voc_size]
        logits_idx:[batch, voc_size]
        
        其中，top_k[batch]对应当前batch采样的k值，有效范围为1≤top_k[batch]≤min(voc_size[batch], 1024)，如果top[k]超出有效范围，则视为跳过当前batch的topk采样阶段，也同样会则跳过当前batch的排序，将logits[batch]直接传入下一模块。

    - topP模块：
        - 内部输入：
            logits_sort:[batch, voc_size]
        - 外部输入：    
            top_p:[batch, 1]

        - 中间结果（不输出）：
            logits_masked:[batch, voc_size]
            logits_idx:[batch, voc_size]
        - 可选输出：
            logits_top_kp_select:[batch, voc_size]

        该模块先对logits_sort按batch进行softmax处理。
        其中，top_p[batch]对应当前batch采样的p值，位于有效范围[0, 1)以内时，对每个batch保留至少1个采样结果。
        当top_p[batch]<1时，为当前batch进行top_p采样，保留至少一个有效logits元素。如果topK模块被跳过，则先使用top_k_guess遍历当前batch：如果对于当前batch的logits_sort按top_k_guess取索引求和后的累积词频超过top_p[batch]，则以logits_sort[batch, top_k_guess[:]]作为logits_masked[batch]，以top_k_guess[:]作为logits_idx[batch, :]；否则对logits_masked:[batch, :]进行全排序和top_p采样。
        如果top_p[batch]≥1，则跳过top-p采样，直接将softmax(logits_sort)作为计算结果。
        如果is_need_logtis=True，则计算logits_top_kp_select[batch, logits_idx[batch, v]] = logits_masked[batch, v]并输出。
    
    - sample模块：
        - 内部输入：
            logits_masked:[batch, voc_size]
        - 外部输入：    
            q:[batch, voc_size]
        - 中间结果（不输出）：
            probs:[batch, voc_size]
            probs_opt:[batch, voc_size]
        - 输出：
            logits_select_idx:[batch, 1]
        
        该模块首先对输入处理，probs[b,:]=softmax(logits_masked[b,:])
        q为有效Tensor时必须与logits尺寸完全一致，此时probs_opt[b, :] = probs[batch, :]/(q[batch, :] + eps)，仅限当前batch对应probs中的有效元素。
        如果q为None，则probs_opt=probs。
    
    - 上述计算模块按每个batch对应的输入参数情况独立使能组合，共计有八种模式。

## 函数原型
```
torch_npu.npu_top_k_top_p_sample(Tensor logits, Tensor top_k, Tensor top_p, Tensor? q=None, float eps=1e-8, bool is_need_logits=False, int top_k_guess=32) -> (Tensor, Tensor)
```

## 参数说明

-   **logits**（`Tensor`）：必选参数，表示待采样的输入词频，目前支持2维，词频索引固定为最后一维。数据类型支持`float16`和`bfloat16`，数据格式支持ND，支持非连续Tensor。
-   **top_k**（`Tensor`）：必选参数，表示每个batch采样的k值，有效范围为1≤top_k[batch]≤min(voc_size[batch], 1024)，目前支持1维。数据类型支持`int32`,数据格式支持ND，支持非连续Tensor。
-   **top_p**（`Tensor`）：必选参数，表示每个batch采样的p值，有效范围为0<top_p[batch]<1，目前支持1维。数据类型和数据格式与logits保持一致，支持非连续Tensor。
-   **q**（`Tensor`）：可选参数，topK-topP采样输出的指数采样矩阵，数据类型支持`float32`，`数据格式支持ND`，`支持非连续Tensor`。
-   **eps**（`float`）：可选属性，在softmax和权重采样中防止除零，默认值1e-8。
-   **is_need_logits**（`bool`）：可选属性，控制logits_top_kp_select的输出条件，默认为False。
-   **top_k_guess**（`int32`）：可选属性，表示每个batch在尝试topP部分遍历采样时的候选logits大小，必须为正整数。

## 输出说明
-   **logits_select_idx**（`Tensor`）：表示经过topK-topP-sample计算流程后，每个batch中词频最大元素max(probs_opt[batch, :])在输入logits中的位置索引。数据类型支持int64，数据格式支持ND。
-   **logits_top_kp_select**（`Tensor`）：表示经过topK-topP计算流程后，输入logtis中剩余未被过滤的logits。数据类型支持float32，数据格式支持ND。

## 约束说明
-   logits、q、logits_top_kp_select的尺寸和维度必须完全一致。
-   logits、top_k、top_p、logits_select_idx除最后一维以外的所有维度必须顺序和大小完全一致。目前logits只能是2维，top_k、top_p、logits_select_idx必须是1维非空Tensor。logits、top_k、top_p不允许空Tensor作为输入，如需跳过相应模块，需按相应规则设置输入。
-   如果需要单独跳过topK模块，请传入[batch, 1]大小的Tensor，并使每个元素均为无效值。
-   如果1024<top_k[batch]<voc_size[batch]，则视为选择当前batch的全部有效元素并跳过topK环节。
-   如果需要单独跳过topP模块，请传入[batch, 1]大小的Tensor，并使每个元素均≥1。
-   如果需要单独跳过sample模块，传入q=None或使用其默认值即可；如需使用sample模块，则必须传入尺寸为[batch, voc_size]的Tensor。

## 支持的芯片型号
Atlas A3 训练系列产品/Atlas A3 推理系列产品
Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 调用示例
```python
>>> logits = torch.from_numpy(np.random.uniform(-2, 2, size=[2, 4])).type(torch.float16).npu()
>>> top_ks = torch.from_numpy(np.random.uniform(1, 2, size=[2, ])).type(torch.int32).npu()
>>> top_ps = torch.from_numpy(np.random.uniform(0.4, 0.5, size=[2, ])).type(torch.float16).npu()
>>> q = torch.from_numpy(np.random.uniform(0.1, 0.5, size=[2, 4])).type(torch.float32).npu()

>>> npu_out_index, logits_top_kp_select = torch_npu.npu_top_k_top_p_sample(logits, top_ks, top_ps, q, 1E-8, True)
>>> print(npu_out_index)
>>> print(logits_top_kp_select)
```
#