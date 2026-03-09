# 为OpPlugin贡献

感谢您考虑为OpPlugin做出贡献！我们欢迎任何形式的贡献，包括错误修复、功能增强、文档改进等，甚至只是反馈。无论您是经验丰富的开发者还是第一次参与开源项目，您的帮助都是非常宝贵的。

您可以通过多种方式支持本项目：

- 通过[issues](https://gitcode.com/Ascend/op-plugin/issues)反馈问题。
- 建议或实现新功能。
- 改进或扩展文档。
- 审查Pull Request并协助其他贡献者。
- 传播项目：在博客文章、社交媒体上分享OpPlugin，或给仓库点个⭐。

## 寻找可贡献的问题

您可以通过查看[issues列表](https://gitcode.com/Ascend/op-plugin/issues)了解项目的发展计划和路线图。

## 贡献流程

-   [贡献者许可协议](#贡献者许可协议)
-   [环境要求](#环境要求)
-   [开发与测试](#开发与测试)

### 贡献者许可协议

在您第一次向 OpPlugin 社区提交代码之前，需要签署 CLA。

对于个人贡献者，详细信息请参考[CLA 在线文档](https://www.mindspore.cn/icla)。

### 环境要求

-  硬件环境请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》
-  需要提前安装CANN开源版本
-  Python 3.10及以上版本
-  CMake 3.11及以上版本

### 开发与测试

1. 先在GitCode平台点击仓库右上角"Fork"按钮，将仓库克隆到个人账户

2. 克隆到本地：

   ```bash
   git clone https://gitcode.com/<your-username>/op-plugin.git
   cd op-plugin
   ```

3. 在个人仓进行代码开发

   代码开发请遵循[代码规范](#代码规范)。

4. 代码测试

   参见[代码测试](#代码测试)。

5. 提交Pull Request

   参见[提交Pull Request流程](#提交pull-request流程)。

### 代码规范

#### Python代码规范

- 遵循PEP 8编码规范
- 使用4个空格进行缩进
- 类名使用大驼峰命名法（如`DataManager`）
- 函数和变量使用小写加下划线命名法（如`parse_data`）
- 添加必要的类型注解和文档字符串

#### C++代码规范

- 遵循项目现有的编码风格
- 使用4个空格进行缩进
- 类名使用大驼峰命名法
- 函数名使用小驼峰命名法
- 添加必要的注释说明复杂逻辑

### 代码测试

#### 运行全量测试

在提交代码前，请确保所有测试通过：
## 构建项目

```bash
# 构建项目
cd ci
./build.sh
```

```bash
# Python单元测试
python run_test.py 
```

#### 添加测试

- 为新功能添加相应的单元测试
- 确保测试覆盖主要逻辑分支
- 测试用例应该具有良好的可读性和维护性
- 测试数据应该放在`test/`目录下的相应位置

### 提交Pull Request流程

#### 提交流程

1. **创建分支**

   ```bash
   git checkout -b feature/<your-feature-name>
   ```

2. **提交更改**

   ```bash
   git add .
   git commit -m "feat: <your feature description>"
   ```

3. **推送到远程仓库**

   ```bash
   git push origin feature/<your-feature-name>
   ```

4. **创建Pull Request**

   在GitCode上创建Pull Request
   根据`.gitcode/PULL_REQUEST_TEMPLATE.md`中的规范模板，完整填写：
   - 合入来源
   - 修改方案
   - 资料变更
   - 接口变更
   - 功能验证
   - CheckList
   确认信息完整准确后提交Pull Request，等待代码审查

## 社区准则

### 行为准则

我们致力于为所有参与者提供一个友好、安全和包容的环境。参与本项目即表示您同意：

- 尊重不同的观点和经验
- 接受建设性的批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

### 沟通渠道

- **Issues**：用于报告Bug、提出功能建议和讨论技术问题
- **Pull Requests**：用于代码审查和讨论具体实现

## 许可证

通过向本项目贡献代码，您同意您的贡献将按照项目的许可证进行授权。详见[LICENSE](../LICENSE)文件。

## 致谢

感谢您为op-plugin做出的贡献。您的努力使这个项目变得更加强大和用户友好。期待您的参与！

---

如有任何疑问或需要帮助，请随时在[issues](https://gitcode.com/Ascend/op-plugin/issues)中提问，或通过其他社区渠道与我们联系。
