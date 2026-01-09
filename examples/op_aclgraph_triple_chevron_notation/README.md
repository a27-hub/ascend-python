# 自定义算子直调并适配aclgraph

## 概述

本样例展示了如何使用PyTorch的torch.library机制注册自定义算子，通过<<<>>>内核调用符调用核函数，并适配aclgraph使用该自定义算子，以简单的Add算子为例，实现两个向量的逐元素相加。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── add_custom.asc       // Add算子实现 & 自定义算子注册
├── add_aclgraph_test.py   // Add算子aclgraph测试demo
├── trig_inplace_custom.asc // 原地三角函数算子实现 & 自定义算子注册
├── inplace_aclgraph_test.py // 原地三角函数aclgraph测试demo
├── CMakeLists.txt           // 编译工程文件
├── README.md                //示例介绍
```

## 算子描述
- Add算子
1. 算子功能：
  Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：
  
  ```
  z = x + y
  ```
2. 算子规格：
  
  <table>
   <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">int</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">int</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">int</td><td align="center">ND</td></tr>
  </tr>
 
  </table>

- 原地三角函数算子
1. 算子功能：
  ```
  sin_out, cos_out, tan_out = sin(x), cos(x), tan(x)
  ```
2. 算子规格：
  <table>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">trig_inplace_custom</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">sin_out</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">cos_out</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  
  </tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输出</td><td align="center">sin_out</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">cos_out</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">tan_out</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
 
  </table>

## 关键代码介绍  

  本样例在*.asc文件中定义了一个名为ascendc_ops的命名空间，并在其中注册了ascendc_add函数。
  
  PyTorch提供`TORCH_LIBRARY_FRAGMENT`宏作为自定义算子注册的核心接口，用于创建并初始化自定义算子库，注册后在Python侧可以通过`torch.ops.namespace.op_name`方式进行调用，例如：
  
  ```c++
  TORCH_LIBRARY_FRAGMENT(ascendc_ops, m) {
      m.def(ascendc_add"(Tensor x, Tensor y) -> Tensor");
  }
  
  
  `TORCH_LIBRARY_IMPL`用于将算子逻辑绑定到特定的DispatchKey（PyTorch设备调度标识）。针对NPU设备，需要将算子实现注册到PrivateUse1这一专属的DispatchKey上，例如：
  
  ```c++
  TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
  {
      m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
  }
  ```
- <<<>>>调用Kernel函数：
  例如，在ascendc_add函数中通过`c10_npu::getCurrentNPUStream()`函数获取当前NPU上的流，并通过内核调用符<<<>>>调用自定义的Kernel函数add_custom，在NPU上执行算子。
  
  ```c++
    add_custom<<<blockDim, nullptr, aclStream>>>(xGm, yGm, zGm, totalLength);
  ```

## Python测试脚本介绍

在add_custom_aclgraph_test.py中，首先通过`torch.ops.load_library`加载生成的自定义算子库，调用注册的ascendc_add函数，并通过对比NPU输出与CPU标准加法结果来验证自定义算子的数值正确性。

- 注册Meta函数：

aclgraph调用算子需要注册Meta函数，本示例在add_custom_aclgraph_test.py开头注册代码如下：

```python
 m = Library("ascendc_ops", "IMPL", "Meta")
  @torch.library.impl(m, "ascendc_add")
  def ascendc_add_meta(a, b):
      return torch.empty_like(a)
```

- aclgraph的调用：
  [示例代码](./add_custom_aclgraph_test.py)中展示了3种aclgraph的使能方式，

1. torch.npu.NPUGraph()
2. torch.npu.make_graphed_callables
3. backend="npugraph_ex"

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。

- 环境安装
  
1. 请参考与您当前使用的版本配套的[《Ascend Extension for PyTorch
   软件安装指南》](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)，获取PyTorch和torch_npu详细的安装步骤，torch_npu支持`backend="npugraph_ex"`需7.3.0版本以上。
2. 根据实际环境安装CANN toolkit包，安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。
3. 运行算子前必须安装社区版CANN legacy包（运行态依赖），若仅编译算子，可跳过本操作。
   
   根据产品型号和环境架构，下载对应`cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run`包，下载链接如下：
   
   - Atlas A2 训练系列产品/Atlas A2 推理系列产品：[legacy x86_64包](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/x86_64/ops/cann-910b-ops-legacy_8.5.0-beta.1_linux-x86_64.run)、[legacy aarch64包](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/aarch64/ops/cann-910b-ops-legacy_8.5.0-beta.1_linux-aarch64.run)。
   - Atlas A3 训练系列产品/Atlas A3 推理系列产品：[legacy x86_64包](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/x86_64/ops/cann-910_93-ops-legacy_8.5.0-beta.1_linux-x86_64.run)、[legacy aarch64包](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/aarch64/ops/cann-910_93-ops-legacy_8.5.0-beta.1_linux-aarch64.run)。
   
   ```bash
   # 确保安装包具有可执行权限
   chmod +x cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run
   # 安装命令
   ./cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
   ```
   
   - \$\{soc\_name\}：表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。
   - \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

- 配置环境变量
  
  请根据当前环境上CANN开发套件包的安装位置，执行如下配置环境变量的命令。
  
    ```bash
    source ${install_path}/ascend-toolkit/set_env.sh
    ```



- 样例执行
  
  ```bash
  mkdir -p build && cd build;     # 创建并进入build目录
  cmake ..;make -j;               # 编译工程
  python3 ../add_custom_aclgraph_test.py   # 执行测试脚本
  ```

执行结果如下，说明精度对比成功。

```bash
Ran * test in **s.
OK
```


