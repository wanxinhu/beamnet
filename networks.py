import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.decoders import EMCAD, LGAG

class EMCADNet(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3,
                 activation='relu6', encoder='resnet34', pretrain=True, pretrained_dir='./pretrained_pth/pvt/',
                 use_as_mscb=True, use_edge=True, use_grab=True, use_stage0=True, use_aaf=True):
        super(EMCADNet, self).__init__()

        self.encoder_type = encoder
        self.use_edge = use_edge  # 记录是否使用边缘增强分支
        self.use_stage0 = use_stage0  # 保存状态

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # ================= 编码器初始化 =================
        if encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels = [512, 256, 128, 64, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels = [512, 256, 128, 64, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels = [2048, 1024, 512, 256, 64]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)
            channels = [2048, 1024, 512, 256, 64]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)
            channels = [2048, 1024, 512, 256, 64]
        else:
            print('Encoder not implemented! Continuing with default encoder resnet34.')
            self.backbone = resnet34(pretrained=pretrain)
            channels = [512, 256, 128, 64, 64]

        print('Model %s created, param count: %d' %
              (encoder + ' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))

        # ================= 解码器初始化 =================
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                             dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation,
                             use_as_mscb=use_as_mscb, use_grab=use_grab, use_stage0=use_stage0, use_aaf=use_aaf)

        print('Model %s created, param count: %d' %
              ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        # ======== 预测头 (Prediction Heads) ========
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

        # 3. 如果关闭了 Stage0，就不需要 out_head0 了
        if self.use_stage0:
            self.out_head0 = nn.Conv2d(channels[4], num_classes, 1)

        if self.use_edge:
            self.edge_extractor = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            )
            # 4. 动态调整 LGAG 的门控通道数：有 d0 用 d0 的通道，没 d0 用 d1 的通道
            gating_channels = channels[4] if self.use_stage0 else channels[3]
            self.edge_lgag = LGAG(F_g=gating_channels, F_l=8, F_int=4, kernel_size=3)
            self.edge_fusion_head = nn.Conv2d(8 + num_classes, num_classes, kernel_size=1)

    def forward(self, x, mode='test'):
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError('Pad height and width to multiples of 32 before inference')
        if x.size()[1] == 1:
            x = self.conv(x)

        # ======== 提取编码器特征 ========
        # 经过修改的 ResNet 现在会稳定返回包含 x0 在内的 5 个特征层级
        x0, x1, x2, x3, x4 = self.backbone(x)

        # ======== 解码器流转 ========
        # 将 x0 传入 skips 列表 [x3, x2, x1, x0]
        dec_outs = self.decoder(x4, [x3, x2, x1, x0])
        # 统一处理前 4 层的深度监督
        d4, d3, d2, d1 = dec_outs[0], dec_outs[1], dec_outs[2], dec_outs[3]
        p4 = F.interpolate(self.out_head4(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p1 = F.interpolate(self.out_head1(d1), scale_factor=4, mode='bilinear', align_corners=False)
        # d4, d3, d2, d1, d0 = dec_outs[0], dec_outs[1], dec_outs[2], dec_outs[3], dec_outs[4]

        # 5. 【核心路由】：根据 Stage 0 的状态，决定最终输出由谁提供
        if self.use_stage0:
            d0 = dec_outs[4]
            p0 = self.out_head0(d0)
            p0_up = F.interpolate(p0, scale_factor=2, mode='bilinear', align_corners=False)
            semantic_feat = d0  # 提供给边缘增强的高阶语义特征
            semantic_pred = p0_up  # 语义预测图
            scale_to_1x1 = 2  # d0 放大的倍数
        else:
            semantic_feat = d1  # 退化为用 d1 (1/4分辨率) 做最终语义
            semantic_pred = p1  # p1 已经是插值到 1/1 的图了
            scale_to_1x1 = 4  # d1 需要放大 4 倍才能和 edge_feat 匹配

        # 执行边缘去噪与增强
        if self.use_edge:
            edge_feat = self.edge_extractor(x)
            # 动态将门控特征放大到原图尺寸
            semantic_feat_up = F.interpolate(semantic_feat, scale_factor=scale_to_1x1, mode='bilinear',
                                             align_corners=False)
            edge_feat_gated = self.edge_lgag(g=semantic_feat_up, x=edge_feat)
            fused_1x1 = torch.cat([semantic_pred, edge_feat_gated], dim=1)
            p_final = self.edge_fusion_head(fused_1x1)
        else:
            p_final = semantic_pred

            # 永远返回 5 个元素的列表，保证 train.py 中的损失函数不崩溃
            # (此时若关闭 Stage 0 且无边缘增强，P[3] 和 P[4] 将都是 p1)
        return [p4, p3, p2, p1, p_final]


# Public name; state-dict keys remain compatible with historical checkpoints.
BEAMNet = EMCADNet
