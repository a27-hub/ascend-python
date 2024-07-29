对于Aclnn适配算子，可通过配置Yaml完成自动代码生成，无需编写代码。

配置字段规则：
-func: api对应的schema，需要与func对应的schema保持一致
-branch：api对应分支，配置选项包括base，v1r11，v2r1, v2r2等，不配置情况下，默认为base
-exec: EXEC_NPU_CMD的参数，如果除aclnnname，其他参数和API参数一致，可只配置aclnnname。
-result：
一个字典，必须配置的选项包括size和dtype，如果包含多个输出结果，可配置成result0, result1等。
可选配置选项，name字段, 如果结果涉及namedTensor特性，可通过此字段配置
-structured_delegate:如果同类的inplace，out-of-place的接口可通过out类接口继承属性，可以填写继承的Aten IR。
