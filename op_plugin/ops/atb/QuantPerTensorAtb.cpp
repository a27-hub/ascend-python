#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/atb_common.h"
#include "op_plugin/utils/custom_functions/atb/op_param_cache.h"
#include <acl/acl.h>

namespace op_api {
using ElewiseParam = atb::infer::ElewiseParam;
void npu_quantpertensor(const at::Tensor &x, const at::Tensor &scale, const at::Tensor &offset, at::Tensor &y, const int64_t elewiseType, const double inputScale, const bool asymmetric, 
    const int64_t inputOffset, const double varAttr, const int64_t outTensorType) 
{
    OpParamCache<ElewiseParam>& elewiseParamCache = OpParamCache<ElewiseParam>::getInstance();
    ElewiseParam elewiseparam;
    auto elewisetype = static_cast<ElewiseParam::ElewiseType>(elewiseType);
    elewiseparam.elewiseType = elewisetype;
    elewiseparam.quantParam.inputScale = inputScale;  
    elewiseparam.quantParam.asymmetric = asymmetric;
    elewiseparam.quantParam.inputOffset = inputOffset;  
    elewiseparam.mulsParam.varAttr = varAttr;
    auto outtensorType = static_cast<aclDataType>(outTensorType);
    elewiseparam.outTensorType = outtensorType;
    
    ParamSetter parametter;
    parametter.Input(x)
                .Input(scale)
                .Input(offset)
                .Output(y);
    auto opReshape = elewiseParamCache.getOperation(elewiseparam, "ElewiseCacheOperation");
    RunAtbCmd(opReshape, parametter, "ElewiseCacheOperation");
    return;
}
} // namespace