# （beta）c10_npu::setCurrentNPUStream

## 定义文件

torch_npu\csrc\core\npu\NPUStream.h

## 函数原型

```
void c10_npu::setCurrentNPUStream(c10_npu::NPUStream stream)
```

## 功能说明

设置当前NPU流，与void c10::CUDA::setCurrentCUDAStream(c10::CUDA::CUDAStream _stream_)相同。
这个函数与当前设备无关，它切换的是传入流所在设备的当前流，将其切换为传入流。

## 参数说明

stream：NPUStream类型，待设置的NPU流。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

