# OpPlugin

## 简介

本项目开发了Ascend Extension for Pytorch（torch_npu）算子插件，为使用PyTorch框架的开发者提供便捷的NPU算子库调用能力。
OpPlugin算子插件的编译、使用依赖昇腾Ascend Extension for PyTorch。因此，在编译OpPlugin之前，需要了解、安装昇腾PyTorch。使用手册可参考昇腾社区[Ascend Extension for Pytorch](https://gitcode.com/ascend/pytorch/blob/master/README.zh.md)。

## 目录结构

关键目录如下:

```ColdFusion
├─docs                             # 文档目录
├─ci                               # 自动化构建与测试脚本目录
├─op_plugin                        # 项目核心目录
│  ├─config                        # 配置管理目录
│  ├─ops                           # 算子实现目录
│  ├─python                        # python绑定目录
├─codegen                          # 代码生成目录
├─examples                         # 示例目录                        
└─test                             # 测试目录
```
