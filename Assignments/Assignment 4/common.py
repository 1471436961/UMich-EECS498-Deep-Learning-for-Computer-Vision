"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    """
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()
        # Replace "pass" statement with your code
        
        ##############################NOTES###################################
        # torch.nn.ModuleDict是PyTorch中的一个容器模块，用于以字典的形式存储子模块。
        # 它继承自nn.Module，可以像其他模块一样被调用、注册参数，并支持模型保存/加载。
        # 与普通字典的区别：
        #     参数跟踪：ModuleDict会跟踪子模块的参数（parameters()），普通字典不会。   
        #     设备移动：调用model.to(device)时，ModuleDict中的模块会自动切换设备。
        ##############################NOTES###################################
        
        # 横向连接 (1x1 conv 调整通道数)
        for level, (name, shape) in zip(["c3", "c4", "c5"], dummy_out_shapes):
            in_channels = shape[1]  # 输入通道数
            self.fpn_params[f"lateral_{level}"] = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
    
        # 输出卷积 (3x3 conv 融合特征)
        for level in ["p3", "p4", "p5"]:
            self.fpn_params[f"output_{level}"] = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        print("FPN layers:", list(self.fpn_params.keys()))
        # 处理 c5 -> p5 (最顶层，无需上采样)
        p5 = self.fpn_params["lateral_c5"](backbone_feats["c5"])
        p5 = self.fpn_params["output_p5"](p5)
        fpn_feats["p5"] = p5
    
        # 处理 c4 -> p4 (融合 p5 上采样)
        p4 = self.fpn_params["lateral_c4"](backbone_feats["c4"])
        p5_upsampled = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4 = p4 + p5_upsampled
        p4 = self.fpn_params["output_p4"](p4)
        fpn_feats["p4"] = p4

        # 处理 c3 -> p3 (融合 p4 上采样)
        p3 = self.fpn_params["lateral_c3"](backbone_feats["c3"])
        p4_upsampled = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = p3 + p4_upsampled
        p3 = self.fpn_params["output_p3"](p3)
        fpn_feats["p3"] = p3

        ##############################NOTES###################################
        # torch.nn.functional.interpolate(
        #     input,               # 输入张量（[B, C, [depth,] H, W]）
        #     size=None,           # 目标尺寸（如 (256, 256)）
        #     scale_factor=None,   # 缩放倍数（如 2.0 表示放大2倍）
        #     mode='nearest',      # 插值模式
        #     align_corners=None,  # 是否对齐角点像素（影响几何对齐，对线性/双线性插值重要）
        #     recompute_scale_factor=None,  # 是否重新计算缩放因子
        #     antialias=False      # 是否启用抗锯齿（仅对 bilinear/bicubic 有效）
        # )
        #                        常见插值模式（mode）
        #    模式	                 适用场景	               特点
        #    nearest	      标签上采样（如分割任务）	 速度快，但边缘锯齿明显
        #    bilinear	        图像/特征图上采样	  平滑输出，适合低分辨率恢复
        #    bicubic	          高质量图像放大	       更平滑，但计算量更大
        #    area	            下采样（缩小图像）	  避免摩尔纹，适合缩小时使用
        #    trilinear	        3D数据（如医学影像）	      在深度方向也插值
        #                           
        #                             注意事项
        # align_corners：       
        #     设为 True 时，输入和输出的角点像素严格对齐（可能影响边缘效果）。
        #     默认 None，在 PyTorch 1.3+ 中与 True 行为一致（推荐显式设置）。                
        # 抗锯齿（antialias）：
        #     仅对 bilinear/bicubic有效，用于抑制上采样时的锯齿伪影（PyTorch1.6+支持）。                
        # 替代方案：
        #     对于常规图像上采样，也可使用转置卷积（nn.ConvTranspose2d），但插值更轻量。
        ##############################NOTES###################################
        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code

        ##############################NOTES###################################
        # torch.meshgrid()用于生成网格坐标，它通过输入的一维坐标向量创建多维网格。
        # torch.meshgrid(*tensors, indexing='ij')
        #     输入：多个一维张量（通常为 arange 生成的序列）。
        #     输出：多个张量（网格化后的坐标矩阵）。        
        #     参数 indexing：        
        #     'ij'：矩阵索引模式（行优先，适用于图像的高度/宽度计算）。        
        #     'xy'：笛卡尔坐标模式（列优先，图形学常用）。
        # 
        # torch.stack()用于沿新维度堆叠张量序列，它将多个张量合并为一个更高维的张量。
        # 常见使用场景：合并多组坐标、批量数据组装、多任务输出合并（如合并分类和回归结果）
        # torch.stack(tensors, dim=0, *, out=None)
        #     tensors：需要堆叠的张量序列（所有张量必须具有相同的形状）。
        #     dim：指定新维度的插入位置（默认 dim=0）。
        #     返回：堆叠后的新张量，维度比输入张量多一维。
        #                       与 torch.cat() 的区别
        #    函数	           行为	              输入要求	        输出维度
        # torch.stack	创建新维度合并张量	所有张量形状必须相同	     dim+1
        #  torch.cat	沿现有维度拼接张量	除拼接维度外其他需相同	 不变
        # 
        # cat使用场景：拼接特征图、合并数据集等
        ##############################NOTES###################################
        
        _, _, H, W = feat_shape

        i = torch.arange(H, dtype=dtype, device=device)
        j = torch.arange(W, dtype=dtype, device=device)
        mesh_i, mesh_j = torch.meshgrid(i, j, indexing='ij')
        
        yc = (mesh_i + 0.5) * level_stride
        xc = (mesh_j + 0.5) * level_stride     

        location_coords[level_name] = torch.stack([yc.reshape(-1), 
                                                   xc.reshape(-1)], dim=1)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
