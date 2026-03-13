import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
#                                                                              
#                      🚀 MODEL: DCSI_UNet (Change Detection)                  
#                       👤 AUTHOR: Caoyv                                        
#                      📄 PAPER: DCSI_UNet 
#           https://ieeexplore.ieee.org/document/11299285                                  
#                                                                              
#   Description: This script belongs to the DCSI_UNet project, designed for     
#   high-precision Remote Sensing Change Detection tasks.                      
#   代码说明：本项目由 Caoyv 开发，旨在实现高精度的遥感图像变化检测任务。            
#                                                                              
################################################################################


""" 
================================================================================
🚀 MODULE: UP (Upsampling)
Description: Provides bilinear interpolation or Transposed Convolution for resolution recovery.
说明：提供双线性插值或转置卷积，用于恢复图像分辨率。
================================================================================
"""
class UP(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(UP, self).__init__()
        if bilinear:
            # Bilinear upsampling / 双线性上采样
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            # Transposed convolution / 转置卷积
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


""" 
================================================================================
🚀 MODULE: Conv (Basic Convolution Unit)
Description: A standard combination of Conv2d, BatchNorm, and ReLU.
说明：标准卷积单元，包含卷积、批归一化和激活函数。
================================================================================
"""
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        assert x.size()[1] == self.inp_dim
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x


""" 
================================================================================
🚀 MODULE: Conv_Block (Residual Convolution Block)
Description: A double convolution block with a residual connection for stable gradients.
说明：带残差连接的双卷积块，有助于梯度稳定。
================================================================================
"""
class Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(out_ch)
        self.ReLU = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # Initial convolution / 初始卷积
        x = self.conv1(x)
        # Residual structure / 残差结构
        return self.ReLU(x + self.BN2(self.conv2(self.ReLU(self.BN1(x)))))


""" 
================================================================================
🚀 MODULE: BasicConv2d
Description: Simplified Convolution-BN wrapper.
说明：简化的卷积-批归一化封装。
================================================================================
"""
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


""" 
================================================================================
🚀 MODULE: Channel_Attention_Module (CAM)
Description: Captures channel relationships using global Average and Max pooling.
说明：通道注意力模块，通过全局平均和最大池化捕捉通道间关系。
================================================================================
"""
class Channel_Attention_Module(nn.Module):
    def __init__(self, in_ch, ratio):
        super(Channel_Attention_Module, self).__init__()
        self.MaxPool = nn.AdaptiveMaxPool2d(1)
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.MLP1 = nn.Conv2d(in_ch, in_ch // ratio, 1)
        self.ReLU = nn.ReLU(inplace=False)
        self.MLP2 = nn.Conv2d(in_ch // ratio, in_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature aggregation / 特征聚合
        Max_out = self.MLP2(self.ReLU(self.MLP1(self.MaxPool(x))))
        Avg_out = self.MLP2(self.ReLU(self.MLP1(self.AvgPool(x))))
        return self.sigmoid(Max_out + Avg_out)


""" 
================================================================================
🚀 MODULE: CMConv (Channel Masked Convolution)
Description: A custom convolution using group masking and dilation for sparse interaction.
说明：通道掩码卷积，利用分组掩码和空洞卷积实现稀疏交互。
================================================================================
"""
class CMConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4, bias=False):
        super(CMConv, self).__init__()
        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation, groups=groups * dilation_set, bias=bias)
        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation, groups=groups * dilation_set, bias=bias)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)

        # Gradient masking hook / 梯度掩码 Hook
        def backward_hook(grad):
            out = grad.clone()
            out[self.mask.bool()] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).byte().cuda() 
        _in_channels = in_ch // (groups * dilation_set)
        _out_channels = out_ch // (groups * dilation_set)
        
        # Generate mask / 生成掩码
        for i in range(dilation_set):
            for j in range(groups):
                self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
                self.mask[((i + dilation_set // 2) % dilation_set + j * groups) * _out_channels: ((i + dilation_set // 2) % dilation_set + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        
        self.conv.weight.data[self.mask.bool()] = 0
        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):
        # Channel splitting and merging / 通道拆分与合并
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift


""" 
================================================================================
🚀 MODULE: SGAM (Similarity-Guided Attention Module)
Description: Fuses dual-temporal features using Gaussian kernels for difference perception.
说明：相似度引导注意力模块，利用高斯核融合双时相特征，增强差异感知。
================================================================================
"""
class SGAM_Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SGAM_Conv_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(out_ch)
        self.ReLU = nn.ReLU(inplace=False)
        self.conv2 = CMConv(out_ch, out_ch, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        return self.ReLU(x + self.BN2(self.conv2(self.ReLU(self.BN1(x)))))
    
class GaoSi_core(nn.Module):
    def __init__(self, in_ch):
        super(GaoSi_core, self).__init__()

    def forward(self, M, A):
        _, _, h, w = A.size()
        q = M.mean(dim=[2, 3], keepdim=True) # Spatial mean / 空间均值
        k = A 
        square = (k - q).pow(2) # Variance calculation / 方差计算
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + 1e-8) + 0.5
        att_weight = nn.Sigmoid()(att_score)
        return att_weight * A

class SGAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SGAM, self).__init__()
        native_ch = out_ch // 2
        self.SGAM_conv = nn.Conv2d(in_ch, native_ch, kernel_size=1)
        self.BN1 = nn.BatchNorm2d(native_ch)
        self.ReLU = nn.ReLU(inplace=True)
        self.GaoSi = GaoSi_core(native_ch)
        self.conv_finally = SGAM_Conv_Block(out_ch, out_ch)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, F1, F2):
        A1 = self.SGAM_conv(F1)
        A2 = self.SGAM_conv(F2)
        A1_wave = self.ReLU(self.BN1(A1))
        A2_wave = self.ReLU(self.BN1(A2))

        M = (A1_wave + A2_wave) * 0.5 # Mutual feature / 交互特征
        A1_hat = self.GaoSi(M, A1)
        A2_hat = self.GaoSi(M, A2)
        result = torch.cat([A1_hat * self.beta + A1, A2_hat * self.beta + A2], dim=1)
        return self.conv_finally(result)


""" 
================================================================================
🚀 MODULE: CGIM (Channel-wise Global Interaction Module)
Description: Deep interaction in channel space using cross-attention mechanisms.
说明：基于交叉注意力机制在通道空间进行深度交互。
================================================================================
"""
class CGIM(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(CGIM, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.key1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.mu = nn.Parameter(torch.zeros(1))
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, F1, F2):
        Q_fuse = self.query(torch.cat([F1, F2], dim=1))
        batch_size, channels, height, width = Q_fuse.shape
        # Reshape for multi-head attention / 针对多头注意力进行重塑
        Q_fuse = Q_fuse.view(batch_size, self.num_heads, -1, height * width).permute(0, 1, 3, 2)

        K1 = self.key1(F1).view(batch_size, self.num_heads, -1, height * width)
        V1 = self.value1(F1).view(batch_size, self.num_heads, -1, height * width)
        K2 = self.key2(F2).view(batch_size, self.num_heads, -1, height * width)
        V2 = self.value2(F2).view(batch_size, self.num_heads, -1, height * width)

        # Cross attention / 交叉注意力
        Att_1 = torch.matmul(self.softmax(K1), self.softmax(Q_fuse))
        X1_wave = torch.matmul(Att_1, V1).view(batch_size, -1, height, width)

        Att_2 = torch.matmul(self.softmax(K2), self.softmax(Q_fuse))
        X2_wave = torch.matmul(Att_2, V2).view(batch_size, -1, height, width)

        return self.conv_cat(torch.cat([self.mu * X1_wave + F1, self.mu * X2_wave + F2], dim=1))


""" 
================================================================================
🚀 MAIN NETWORK: DCSI_UNet
Description: Dual-path encoder with CGIM/SGAM and IAM for change detection tasks.
说明：带 CGIM/SGAM 和 IAM 模块的双路径编码器 U-Net，用于变化检测任务。
================================================================================
"""
class DCSI_UNet(nn.Module):
    def __init__(self, pretrained=False):
        super(DCSI_UNet, self).__init__()
        torch.nn.Module.dump_patches = True

        n1 = 32
        filters = [n1 * 1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.MaxPooling = nn.MaxPool2d(2, 2)

        # Siamese Encoders / 孪生编码器
        self.En_ResConv1_1 = Conv_Block(3, filters[0]) 
        self.En_ResConv1_2 = Conv_Block(filters[0], filters[1]) 
        self.En_ResConv1_3 = Conv_Block(filters[1], filters[2]) 
        self.En_ResConv1_4 = Conv_Block(filters[2], filters[3])

        self.En_ResConv2_1 = Conv_Block(3, filters[0])
        self.En_ResConv2_2 = Conv_Block(filters[0], filters[1])
        self.En_ResConv2_3 = Conv_Block(filters[1], filters[2])
        self.En_ResConv2_4 = Conv_Block(filters[2], filters[3])

        # Interaction Modules / 交互模块 (CGIM & SGAM)
        self.CGIM1 = CGIM(filters[0], filters[0], 4)
        self.CGIM2 = CGIM(filters[1], filters[1], 4)
        self.CGIM3 = CGIM(filters[2], filters[2], 4)
        self.CGIM4 = CGIM(filters[3], filters[3], 4)

        self.SGAM1 = SGAM(filters[0], filters[0])
        self.SGAM2 = SGAM(filters[1], filters[1])
        self.SGAM3 = SGAM(filters[2], filters[2])
        self.SGAM4 = SGAM(filters[3], filters[3])

        # Decoders / 解码器结构
        self.De_up1 = UP(filters[1])
        self.De_up2 = UP(filters[2])
        self.De_up3 = UP(filters[3])
        
        self.De_ResConv1_1 = Conv_Block(filters[0] + filters[1], filters[0]) 
        self.De_ResConv1_2 = Conv_Block(filters[1] + filters[2], filters[1])
        self.De_ResConv1_3 = Conv_Block(filters[2] + filters[3], filters[2])
        self.De_ResConv1_4 = Conv_Block(filters[3], filters[3])

        self.De_ResConv2_1 = Conv_Block(filters[0] + filters[1], filters[0])
        self.De_ResConv2_2 = Conv_Block(filters[1] + filters[2], filters[1])
        self.De_ResConv2_3 = Conv_Block(filters[2] + filters[3], filters[2])
        self.De_ResConv2_4 = Conv_Block(filters[3], filters[3])

        # IAM Modules (Inter-stage Attention Module) / 级间注意力模块
        self.IAM_up2 = UP(2 * filters[1], True)
        self.IAM_up3 = UP(2 * filters[2], True)
        self.IAM_up4 = UP(2 * filters[3], True)

        self.IAM_Conv1_1 = BasicConv2d(2 * filters[0], filters[0], kernel_size=1)
        self.IAM_Conv2_1 = BasicConv2d(2 * filters[1], filters[1], kernel_size=1)
        self.IAM_Conv3_1 = BasicConv2d(2 * filters[2], filters[2], kernel_size=1)
        self.IAM_Conv4_1 = BasicConv2d(2 * filters[3], filters[3], kernel_size=1)

        self.IAM_Channel2 = Channel_Attention_Module(filters[1], 4)
        self.IAM_Channel3 = Channel_Attention_Module(filters[2], 8)
        self.IAM_Channel4 = Channel_Attention_Module(filters[3], 16)

        self.IAM_Conv2_2 = nn.Conv2d(filters[1], filters[1], 3, padding=1)
        self.IAM_Conv3_2 = nn.Conv2d(filters[2], filters[2], 3, padding=1)
        self.IAM_Conv4_2 = nn.Conv2d(filters[3], filters[3], 3, padding=1)

        # Adaptive Prediction Heads / 自适应预测头
        self.Pre_up = UP(filters[0], True)
        self.a1 = nn.Parameter(torch.ones(1))
        self.a2 = nn.Parameter(torch.ones(1))
        self.a3 = nn.Parameter(torch.zeros(1))

        self.Pre_conv1 = nn.Sequential(Conv(filters[0], 16, 3, bn=True), Conv(16, 2, 3, bn=False, relu=False))
        self.Pre_conv2 = nn.Sequential(Conv(filters[0], 16, 3, bn=True), Conv(16, 2, 3, bn=False, relu=False))
        self.Pre_conv3 = nn.Sequential(Conv(filters[0], 16, 3, bn=True), Conv(16, 2, 3, bn=False, relu=False))

        # Weight Initialization / 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, Img1, Img2):
        # Feature Extraction / 特征提取
        F1_1 = self.MaxPooling(self.En_ResConv1_1(Img1))
        F1_2 = self.MaxPooling(self.En_ResConv1_2(F1_1))
        F1_3 = self.MaxPooling(self.En_ResConv1_3(F1_2))
        F1_4 = self.MaxPooling(self.En_ResConv1_4(F1_3))

        F2_1 = self.MaxPooling(self.En_ResConv2_1(Img2))
        F2_2 = self.MaxPooling(self.En_ResConv2_2(F2_1))
        F2_3 = self.MaxPooling(self.En_ResConv2_3(F2_2))
        F2_4 = self.MaxPooling(self.En_ResConv2_4(F2_3))

        # Interaction & Difference / 交互与差异计算
        Fc1 = self.CGIM1(F1_1, F2_1); Fc2 = self.CGIM2(F1_2, F2_2)
        Fc3 = self.CGIM3(F1_3, F2_3); Fc4 = self.CGIM4(F1_4, F2_4)

        Fs1 = self.SGAM1(F1_1, F2_1); Fs2 = self.SGAM2(F1_2, F2_2)
        Fs3 = self.SGAM3(F1_3, F2_3); Fs4 = self.SGAM4(F1_4, F2_4)

        # Decoding / 解码
        Yc4 = self.De_ResConv1_4(Fc4)
        Yc3 = self.De_ResConv1_3(torch.cat([Fc3, self.De_up3(Yc4)], dim=1))
        Yc2 = self.De_ResConv1_2(torch.cat([Fc2, self.De_up2(Yc3)], dim=1))
        Yc1 = self.De_ResConv1_1(torch.cat([Fc1, self.De_up1(Yc2)], dim=1))

        Ys4 = self.De_ResConv2_4(Fs4)
        Ys3 = self.De_ResConv2_3(torch.cat([Fs3, self.De_up3(Ys4)], dim=1))
        Ys2 = self.De_ResConv2_2(torch.cat([Fs2, self.De_up2(Ys3)], dim=1))
        Ys1 = self.De_ResConv2_1(torch.cat([Fs1, self.De_up1(Ys2)], dim=1))

        # IAM stage / IAM 级联处理
        Fg_4 = torch.cat([Fc4, Fs4], dim=1); Fg_3 = torch.cat([Fc3, Fs3], dim=1)
        Fg_2 = torch.cat([Fc2, Fs2], dim=1); Fg_1 = torch.cat([Fc1, Fs1], dim=1)
        
        Fg_wave_4 = self.IAM_Conv4_1(self.IAM_up4(Fg_4))
        Fg_hat_3 = F.relu((Fg_3 + self.IAM_Channel4(Fg_wave_4)) * self.IAM_Conv4_2(Fg_wave_4), inplace=True)
        Fg_wave_3 = self.IAM_Conv3_1(self.IAM_up3(Fg_hat_3))
        Fg_hat_2 = F.relu((Fg_2 + self.IAM_Channel3(Fg_wave_3)) * self.IAM_Conv3_2(Fg_wave_3), inplace=True)
        Fg_wave_2 = self.IAM_Conv2_1(self.IAM_up2(Fg_hat_2))
        Fg_hat_1 = F.relu((Fg_1 + self.IAM_Channel2(Fg_wave_2)) * self.IAM_Conv2_2(Fg_wave_2), inplace=True)
        Yg = self.IAM_Conv1_1(Fg_hat_1) 

        # Final predictions / 最终预测结果
        P1 = self.a1 * self.Pre_conv1(self.Pre_up(Yc1))
        P2 = self.a2 * self.Pre_conv2(self.Pre_up(Ys1))
        P3 = self.a3 * self.Pre_conv3(self.Pre_up(Yg))

        return P1, P2, P3

# Testing Entry / 测试入口
if __name__ == "__main__":
    x1 = torch.rand(1, 3, 256, 256).cuda()
    x2 = torch.rand(1, 3, 256, 256).cuda()
    Net = DCSI_UNet().cuda()
    
    # Calculate GFLOPs & Parameters / 计算 FLOPs 与参数量
    from thop import profile
    flops, params = profile(Net, inputs=(x1, x2))
    print(f"Model FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Model Parameters: {params / 1e6:.2f} M")