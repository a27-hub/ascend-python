#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include "atb_common.h"

using namespace std;


static atb::Context* msContext = nullptr;


at::Tensor FormatTrans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        return at_npu::native::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}


atb::Tensor AtTensor2Tensor(const at::Tensor atTensor)
{
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, ACL_BOOL},   {at::ScalarType::Byte, ACL_UINT8},
        {at::ScalarType::Char, ACL_INT8},   {at::ScalarType::Half, ACL_FLOAT16},
        {at::ScalarType::Float, ACL_FLOAT}, {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64}, {at::ScalarType::BFloat16, ACL_BF16},
    };

    TORCH_CHECK(atTensor.is_contiguous(), "atTensor is not contiguous");
    atb::Tensor tensor;
    tensor.desc.format = ACL_FORMAT_ND;
    if (atTensor.device().type() == at::kCPU) {
        tensor.hostData = atTensor.data_ptr();
    } else {
        tensor.deviceData = atTensor.data_ptr();
    }

    tensor.desc.shape.dimNum = atTensor.sizes().size();
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        tensor.desc.shape.dims[i] = atTensor.sizes()[i];
    }

    auto it = dtypeMap.find(atTensor.scalar_type());
    TORCH_CHECK(it != dtypeMap.end(), "not support dtype:");
    tensor.desc.dtype = it->second;

    tensor.dataSize = atb::Utils::GetTensorSize(tensor);

    return tensor;
}


void RunAtbCmd(atb::Operation *op, const ParamSetter &paramsetter, const std::string &name)
{
    auto acl_call = [=]() -> int {
        auto contextPtr = GetContext();
        uint64_t workspaceSize = OperationSetup(paramsetter.variantPack, op, contextPtr);
        auto workspaceTensor = GetWorkspaceTensor(workspaceSize);
        const void *workspacePtr = workspaceTensor.storage().data();
        auto st = op->Execute(paramsetter.variantPack, (uint8_t *)workspacePtr, workspaceSize, contextPtr);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(name, acl_call);
}


ParamSetter& ParamSetter::Input(const at::Tensor &tensor)
{
    if (!tensor.defined()) {
        variantPack.inTensors.push_back(atb::Tensor());
        return *this;
    }
    at::Tensor newTensor = tensor;
    if (torch_npu::utils::is_npu(newTensor)) {
        newTensor = FormatTrans(tensor);
    }

    if(!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }
    auto AtTensor = AtTensor2Tensor(newTensor);

    variantPack.inTensors.push_back(AtTensor);
    return *this;
}


ParamSetter& ParamSetter::Input(const c10::optional<at::Tensor> &tensor)
{
    if (!tensor.has_value()) {
        variantPack.inTensors.push_back(atb::Tensor());
        return *this;
    }
    return Input(tensor.value());
}


ParamSetter& ParamSetter::Output(at::Tensor &output)
{
    auto AtTensor = AtTensor2Tensor(output);
    variantPack.outTensors.push_back(AtTensor);
    return *this;
}


uint64_t OperationSetup(atb::VariantPack variantPack, atb::Operation *operation, atb::Context* contextPtr)
{
    uint64_t workspaceSize = 0;
    atb::Status status = operation->Setup(variantPack, workspaceSize, contextPtr);
    TORCH_CHECK(status == 0, "setup failed!");
    return workspaceSize;
}


at::Tensor GetWorkspaceTensor(uint64_t workspaceSize)
{
    
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    // at::Tensor workspaceTensor = at::empty(at::IntArrayRef(workspaceSize), options.dtype(at::kByte));
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false); 
    at::Tensor workspaceTensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspaceSize, acl_stream);
    return workspaceTensor;
}


atb::Context* GetContext()
{        
    if (msContext == nullptr) {
        auto status = atb::CreateContext(&msContext);
        TORCH_CHECK(status == 0, "create context failed!");
        int32_t devId = 0;
        aclrtGetDevice(&devId);
        aclrtStream stream = c10_npu::getCurrentNPUStream(devId).stream(false);
        TORCH_CHECK(stream != nullptr, "get current stream failed");
        msContext->SetExecuteStream(stream);
    }
    return msContext;
}
