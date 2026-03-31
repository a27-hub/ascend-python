aclops目录（检测/图像类）：
NMS系列：NmsV4, NmsWithMask, NmsRotated, BatchNMS
IOU系列：Iou, Diou, Ciou, Giou, RotatedIou, RotatedOverlaps
BoundingBox系列：BoundingBoxEncode, BoundingBoxDecode, RotatedBox系列
ROI系列：RoiAlign, PsRoiPooling
Upsample系列：UpsampleBilinear, UpsampleBicubic, UpsampleNearest, UpsampleTrilinear
图像处理：CropAndResize, DecodeJpeg, ImageNormalize, ImgToTensor
GridSampler系列
AnchorResponseFlags, GridAssignPositive
YoloBoxesEncode
DeformableConv2d
opapi目录（对应算子）：
NMS系列
IOU系列
RoiAlign系列
Upsample系列（全部）
GridSampler系列
DeformableConv2d
AffineGridGenerator
examples目录（全部）：
aclnn_extension, cpp_extension, framwork_cpp_extension, kernel_extension_aclgraph