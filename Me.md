1.	【Modellink】掌握兼容性方案性能精度调试调优，完成LLama32_1b_8k、Llama2_7b、tune-llama31-8b-8k-full、MM-opensoraplan1.3_2v_pretrain、DeepSeek r1-zero qwen25-7b等模型精度强对齐，性能拆解定位到具体责任方
-Wiki: Qwen 14B精度对齐
-Wiki: llama-13B精度对齐
-Wiki: LLama32_1b_8k精度对齐
-Wiki: LLama2_7b精度对齐 
-Wiki: tune-llama31-8b-8k-full精度对齐
-Wiki: MM-opensoraplan1.3_2v_pretrain精度对齐
-Wiki: MindSpeed-RL r1-qwen-7b转测自验
-Wiki: 性能优化 
2.	【兼容性方案】作为Qwen-GRPO模型FO，基于MindSpeed-RL仓进行7B、32B模型2.0.0分支、master分支、共仓方案、共卡方案的精度内存对齐工作，主导交付对应的迭代需求，按期转测
-Wiki: MindSpeed-RL共卡 deepseekR1-Qwen25-7B大模型自验测试报告模板（RL类）

3.	【门禁用例】主导完成Qwen-GRPO门禁用例上库（MindSpeed-Core-MS、msadapter、联合流水线、mindspore）
4.	【自动化测试脚本】完成Qwen-GRPO自动化测试用例并交付测试
【软件贡献】
1.	输出精度性能内存定位6+Wiki文档
2.	蓝区提交代码2.8k+
3.	5+问题单清零
【团队贡献】
公共事务（招聘、新员工培养等）/赋能培训/工具孵化/效率提升等等
1.	完成3次团队内性能/精度/功能定位经验分享。
2.	完成1次团队内强化学习GRPO赋能分享

1.	【Modellink对接】
1）	完成Qwen14B、llama-13B、LLama32_1b_8k、Llama2_7b、tune-llama31-8b-8k-full、MM-opensoraplan1.3_2v_pretrain等模型适配，精度强对齐，性能拆解优化，需求按时转测，输出7+wiki经验文档
2）	tune-llama31-8b-8k-full内存定位对齐PTA，识别兼容层hook无梯度返回、框架侧matmul释放时间晚等问题，优化静态显存7个G，动态显存4个G。

2.	【兼容性专项】完成DeepSeek r1-zero qwen25-7b精度对齐，需求转测。基于MindSpeed-RL仓进行GRPO qwen25-7B、32B模型共卡方案精度内存对齐，云上训练等工作，输出7+经验文档
1）	识别定位并规避解决glm4 32b网络内存clone不释放、RL中tensor.copy_内存拷贝冗余、GRPO场景进程结束无析构导致开启mem tracker失败、GRPO训练compute reward卡死、A3机器不支持FA算子反向梯度无法对齐PTA、框架不支持cpu tensor和numpy内存共享等问题 
2）	云上搭建多机环境，复现GRPO qwen25-7B训练曲线，输出云上多机终端调试经验文档1篇
3）	5+问题单清零

3.	【MM扫网】
1）	基于MindSpeed-MM仓主导完成13+模型规格精度强对齐，需求网络wan2.1权重转换、特征提取、精度对齐并按时转测，输出11+wiki经验文档
2）	MM扫网生成类网络patch分析，识别兼容层rturediv，matmul，linear等算子精度问题适配对齐pta，适配兼容层reshape，maxpool3d等缺失或不兼容接口
3）	定位wan2.1 1.3 lora内存问题，识别框架侧dense算子，inplaceadd算子问题，裁剪网络释放1000MB对齐pta，
4）	清零10+问题单

4.	【门禁用例】主导完成Qwen-GRPO门禁用例上库（MindSpeed-Core-MS、msadapter、联合流水线、mindspore）

5.	【自动化测试脚本】完成Qwen-GRPO自动化测试用例并交付测试

【软件贡献】
1.	输出精度性能内存定位25+Wiki文档
2.	蓝区提交代码8.3k+
3.	全年15+问题单清零

