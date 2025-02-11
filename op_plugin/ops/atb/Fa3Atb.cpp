#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/atb_common.h"
#include "op_plugin/utils/custom_functions/atb/op_param_cache.h"
#include <acl/acl.h>

using namespace std;

namespace op_api {
using PagedAttentionParam = atb::infer::PagedAttentionParam;
void npu_fa3(const at::Tensor &query, const at::Tensor &keyCache, const at::Tensor &valueCache, const int64_t kvHeadNum, const int64_t headNum, const double qkScale, const at::Tensor &blockTables, const at::Tensor &contextLens, 
    const int64_t maskType, const bool batchRunStatusEnable, const int64_t quantType, const int64_t outDataType, const bool hasQuantOffset, const int64_t compressType, const int64_t calcType, const int64_t scaleType, 
    const int64_t inputLayout, const int64_t mlaVHeadSize, const at::Tensor &kDescale, const at::Tensor &vDescale, at::Tensor & attnOut) 
{
    OpParamCache<PagedAttentionParam>& pagedAttentionParamCache = OpParamCache<PagedAttentionParam>::getInstance();
    PagedAttentionParam pagedparam;
    pagedparam.headNum = headNum;
    pagedparam.qkScale = qkScale;
    pagedparam.kvHeadNum = kvHeadNum;
    auto masktype = static_cast<PagedAttentionParam::MaskType>(maskType);
    pagedparam.maskType = masktype;
    pagedparam.batchRunStatusEnable = batchRunStatusEnable;
    auto quanttype = static_cast<PagedAttentionParam::QuantType>(quantType);
    pagedparam.quantType = quanttype;
    auto outdataType = static_cast<aclDataType>(outDataType);
    pagedparam.outDataType = outdataType; 
    pagedparam.hasQuantOffset = hasQuantOffset;
    auto compresstype = static_cast<PagedAttentionParam::CompressType>(compressType);
    pagedparam.compressType = compresstype;
    auto calctype = static_cast<PagedAttentionParam::CalcType>(calcType);
    pagedparam.calcType = calctype;
    auto scaletype = static_cast<PagedAttentionParam::ScaleType>(scaleType);
    pagedparam.scaleType = scaletype;
    auto inputlayout = static_cast<atb::infer::InputLayout>(inputLayout);
    pagedparam.inputLayout = inputlayout;
    auto mlavHeadSize = static_cast<uint32_t>(mlaVHeadSize);
    pagedparam.mlaVHeadSize = mlavHeadSize;
    // pagedparam.headNum = headNum;
    // pagedparam.qkScale = qkScale;
    // pagedparam.kvHeadNum = kvHeadNum;
    // pagedparam.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
    // pagedparam.batchRunStatusEnable = false;
    // //pagedparam.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_UNDEFINED;
    // //pagedparam.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE;
    // //pagedparam.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE;
    // auto quanttype = static_cast<atb::infer::PagedAttentionParam::QuantType>(quantType);
    // pagedparam.quantType = quanttype;
    // pagedparam.outDataType = ACL_DT_UNDEFINED;  
    // pagedparam.hasQuantOffset = false;
    // pagedparam.compressType = atb::infer::PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
    // pagedparam.calcType = atb::infer::PagedAttentionParam::CALC_TYPE_UNDEFINED;
    // pagedparam.scaleType = atb::infer::PagedAttentionParam::SCALE_TYPE_TOR;
    // pagedparam.inputLayout = atb::infer::TYPE_BSND;
    // pagedparam.mlaVHeadSize = 0;  
    ParamSetter paramsetter;
    //if (quanttype == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE || quanttype == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE){
    paramsetter.Input(query)
            .Input(keyCache)
            .Input(valueCache)
            .Input(blockTables)
            .Input(contextLens)
            .Input(kDescale)
            //.Input(kOffset)
            .Input(vDescale)
            //.Input(vOffset)
            .Output(attnOut);
    //}
    auto opPaged = pagedAttentionParamCache.getOperation(pagedparam, "PagedAttentionOperation");
    RunAtbCmd(opPaged, paramsetter, "PagedAttentionOperation");

    return;
}
}