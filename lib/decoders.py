import torch
import torch.nn as nn
from functools import partial

import math
def trunc_normal_tf_(tensor, std=.02):
    # TensorFlow convention: truncate a unit normal at +/-2, then scale.
    nn.init.trunc_normal_(tensor, std=1., a=-2., b=2.)
    with torch.no_grad():
        tensor.mul_(std)
    return tensor


def named_apply(fn, module):
    # timm's default is depth-first, include_root=False.
    for name, child in module.named_children():
        named_apply(fn, child)
        fn(child, name)
    return module


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


# =====================================================================
# 不对称深度可分离卷积模块 (Asymmetric Depth-wise Convolution)
# 包含消融开关：use_as_mscb
# =====================================================================
class AsymmetricDWConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, use_as_mscb=True):
        super(AsymmetricDWConv, self).__init__()
        self.kernel_size = kernel_size
        self.use_as_mscb = use_as_mscb

        if kernel_size == 1 or not use_as_mscb:
            # 如果核大小为1，或者关闭了不对称卷积(消融实验)，则使用标准的方形深度可分离卷积
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, kernel_size // 2, groups=in_channels,
                                  bias=False)
        else:
            pad = kernel_size // 2
            # 水平流: 1 x K 卷积
            self.conv_h = nn.Conv2d(in_channels, in_channels, (1, kernel_size), stride, (0, pad), groups=in_channels,
                                    bias=False)
            # 垂直流: K x 1 卷积
            self.conv_v = nn.Conv2d(in_channels, in_channels, (kernel_size, 1), stride, (pad, 0), groups=in_channels,
                                    bias=False)

    def forward(self, x):
        if self.kernel_size == 1 or not self.use_as_mscb:
            return self.conv(x)
        else:
            # 将水平和垂直方向提取到的线状特征进行逐元素相加融合
            return self.conv_h(x) + self.conv_v(x)


#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True, use_as_mscb=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                AsymmetricDWConv(self.in_channels, kernel_size, stride, use_as_mscb=use_as_mscb),  # 传递消融开关
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    包含消融开关：use_as_mscb
    """

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6', use_as_mscb=True):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        self.use_as_mscb = use_as_mscb  # 记录消融开关

        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        # 1. Split: 多尺度特征提取
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel, use_as_mscb=self.use_as_mscb)

        # =====================================================================
        # SKU (Selective Kernel Unit) 注意力机制组件
        # =====================================================================
        if self.use_as_mscb:
            self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

            # 设置 SKU 机制的标准超参数
            reduction = 16  # 降维比例
            L = 32  # 最小通道数阈值

            # 确定压缩后的通道维度 d
            d = max(self.ex_channels // reduction, L)

            # Fuse: 生成紧凑的引导特征 z
            self.fc1 = nn.Sequential(
                nn.Conv2d(self.ex_channels, d, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True)
            )

            # Select: 为 n_scales 个分支的每个通道生成权重
            self.fc2 = nn.Conv2d(d, self.ex_channels * self.n_scales, 1, bias=False)
        # =====================================================================

        self.combined_channels = self.ex_channels

        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)

        # msdc_outs 是一个列表，包含 [1x1特征, 3x3特征, 5x5特征]
        msdc_outs = self.msdc(pout1)

        if self.use_as_mscb:
            # =====================================================================
            # SKU 的自适应特征聚合
            # =====================================================================
            U = sum(msdc_outs)
            s = self.gap(U)
            z = self.fc1(s)

            attn_weights = self.fc2(z)  # shape: [B, C * n_scales, 1, 1]
            batch_size = attn_weights.shape[0]

            attn_weights = attn_weights.view(batch_size, self.n_scales, self.ex_channels, 1, 1)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)

            dout = 0
            for i, dwout in enumerate(msdc_outs):
                dout += dwout * attn_weights[:, i, :, :, :]
        else:
            # 消融：如果不使用 SKU，则退化为简单的特征相加 (Baseline模式)
            dout = sum(msdc_outs)

        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


#   Multi-scale convolution block (MSCB)
def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6', use_as_mscb=True):
    """
    create a series of multi-scale convolution blocks.
    """
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation, use_as_mscb=use_as_mscb)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation, use_as_mscb=use_as_mscb)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv


#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


#   Large-kernel grouped attention gate (LGAG)
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

    #   Spatial attention block (SAB)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


# =====================================================================
# Advanced Attention Fusion (AAF)
# =====================================================================
# =====================================================================
# Advanced Attention Fusion (AAF)
# =====================================================================
class AdvancedAttentionFusion(nn.Module):
    """
    高级注意力融合模块 / 原版基线融合模块 (受控于 use_aaf)
    """

    def __init__(self, channels, lgag_ks=3, use_aaf=True):
        super(AdvancedAttentionFusion, self).__init__()
        self.use_aaf = use_aaf  # 记录 AAF 消融开关状态

        # 1. LGAG：利用解码器特征引导编码器特征进行空间定位
        self.lgag = LGAG(F_g=channels, F_l=channels, F_int=channels // 2, kernel_size=lgag_ks, groups=channels // 2)

        # 2. CAB：对特征进行通道去噪
        self.cab = CAB(in_channels=channels)

        # 3. SAB：对特征进行空间注意力强化
        self.sab = SAB(kernel_size=7)

    def forward(self, x_enc, x_dec_up):
        if self.use_aaf:
            # 【你的创新版本 (AAF)】：双流并行注意力机制
            # --- 编码器流 (Skip Connection) ---
            x_gated = self.lgag(g=x_dec_up, x=x_enc)
            enc_attended = self.cab(x_gated) * x_gated

            # --- 解码器流 (Upsampled Feature) ---
            dec_attended = self.sab(x_dec_up) * x_dec_up

            # --- 融合 ---
            return enc_attended + dec_attended
        else:
            # 【原版基线 (Baseline)】：单流串行注意力机制 (LGAG -> Add -> CAB -> SAB)
            x_gated = self.lgag(g=x_dec_up, x=x_enc)
            fused = x_gated + x_dec_up

            cab_out = self.cab(fused) * fused
            sab_out = self.sab(cab_out) * cab_out

            return sab_out


# =====================================================================
# EMCAD (解码器主网络)
# =====================================================================
class EMCAD(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64, 32], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu6', use_as_mscb=True, use_grab=True, use_stage0=True, use_aaf=True):
        super(EMCAD, self).__init__()
        eucb_ks = 3
        self.use_grab = use_grab
        self.use_stage0 = use_stage0
        self.use_aaf = use_aaf # 新增 AAF 开关

        # ================= 瓶颈层 (Bottleneck - 最底层) =================
        self.cab4 = CAB(channels[0])
        self.sab = SAB()
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation, use_as_mscb=use_as_mscb)

        # ================= Stage 3 =================
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.aaf3 = AdvancedAttentionFusion(channels=channels[1], lgag_ks=lgag_ks, use_aaf=self.use_aaf)
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation, use_as_mscb=use_as_mscb)

        # ================= Stage 2 =================
        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.aaf2 = AdvancedAttentionFusion(channels=channels[2], lgag_ks=lgag_ks, use_aaf=self.use_aaf)
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation, use_as_mscb=use_as_mscb)

        # ================= Stage 1 =================
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.aaf1 = AdvancedAttentionFusion(channels=channels[3], lgag_ks=lgag_ks, use_aaf=self.use_aaf)
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation, use_as_mscb=use_as_mscb)

        if self.use_stage0:
            self.eucb0 = EUCB(in_channels=channels[3], out_channels=channels[4], kernel_size=3, stride=1)
            self.aaf0 = AdvancedAttentionFusion(channels=channels[4], lgag_ks=lgag_ks, use_aaf=self.use_aaf)
            self.mscb0 = MSCBLayer(channels[4], channels[4], n=1, stride=1, kernel_sizes=kernel_sizes,
                                   expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                                   activation=activation, use_as_mscb=use_as_mscb)
    # forward 函数保持不变

    def forward(self, x, skips):
        # skips 包含 4 个元素: [x3, x2, x1, x0]

        # ================= Bottleneck =================
        if self.use_grab:
            # 【你的创新版本 (GRAB)】：特征提取前置 + 全局跨模块残差
            d4_feat = self.mscb4(x)
            d4_att = self.cab4(d4_feat) * d4_feat
            d4_att = self.sab(d4_att) * d4_att
            d4 = x + d4_att
        else:
            # 【原版基线 (Baseline)】：注意力先行 + 仅局部残差 MSCB(SAB(CAB(x)))
            d4_cab = self.cab4(x) * x
            d4_sab = self.sab(d4_cab) * d4_cab
            d4 = self.mscb4(d4_sab)

        # ================= Stage 3 =================
        d3_up = self.eucb3(d4)
        d3_fused = self.aaf3(x_enc=skips[0], x_dec_up=d3_up)
        d3 = self.mscb3(d3_fused)

        # ================= Stage 2 =================
        d2_up = self.eucb2(d3)
        d2_fused = self.aaf2(x_enc=skips[1], x_dec_up=d2_up)
        d2 = self.mscb2(d2_fused)

        # ================= Stage 1 =================
        d1_up = self.eucb1(d2)
        d1_fused = self.aaf1(x_enc=skips[2], x_dec_up=d1_up)
        d1 = self.mscb1(d1_fused)

        # 3. 动态控制返回值
        if self.use_stage0:
            d0_up = self.eucb0(d1)
            d0_fused = self.aaf0(x_enc=skips[3], x_dec_up=d0_up)  # 与 x0 融合
            d0 = self.mscb0(d0_fused)
            return [d4, d3, d2, d1, d0]  # 返回 5 层
        else:
            return [d4, d3, d2, d1]  # 消融模式：只返回 4 层
        # ================= Stage 0 =================
