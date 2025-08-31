// Copyright (c) 2023 Huawei Technologies Co., Ltd. All rights reserved.

#include "op_plugin/OpApiInterface.h"

namespace op_api {
at::Tensor view_as_real(const at::Tensor& self)
{
    return at::native::view_as_real(self);
}

at::Tensor view_as_complex(const at::Tensor& self)
{
    return at::native::view_as_complex(self);
}

}
