1、算子接口：  
- func: npu_grouped_dynamic_block_quant(Tensor x, Tensor group_list, *, float min_scale=0.0, str round_mode="rint", int          dst_type=291, int row_block_size=1, int col_block_size=128, int group_list_type=0) -> (Tensor, Tensor)
    op_api: [v2.1, newest]

2、输出 shape 推导，输出y的shape和x的shape一样；
scale的shape由x的shape推导而来，当group_list_type等于0时方能生效，
当scale_shape为二维时，
scale_shape第0维的大小等于scale_shape第0维的大小 / row_block_size + group_list的shape
scale_shape第1维的大小等于 (scale_shape第1维的大小 / col_block_size)并向上取整
当scale_shape为三维时，
scale_shape第1维的大小等于scale_shape第0维的大小 / row_block_size + group_list的shape
scale_shape第2维的大小等于 (scale_shape第2维的大小 / col_block_size)并向上取整   

3、第一个输出的数据类型由dst_type决定，其输出类型可以为FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8
第二个输出类型为FLOAT32

4、目标芯片为Ascend950

