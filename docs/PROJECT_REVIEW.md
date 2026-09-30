# BEAM-Net 发布前检查稿

## 代码来源与差异

| 目录 | 作用判断 | 关键特征 |
|---|---|---|
| `D:\projects\retina_seg` | 早期基础版 | 结构接近 EMCAD 原始实现；`test.py` 默认 CHASE_DB1；含动态圆形 FOV 处理实验代码。 |
| `D:\projects\retina_seg -bianyuanzengqiang` | 边缘增强实验版 | 加入四方向 Sobel 权重图和若干消融开关；这些内容不属于论文最终模型。 |
| `D:\projects\xiaorongshiyan` | 最完整实验汇总版 | 保留论文所需的多模块解码器、AAF、边缘预测头、clDice、深监督、可视化和实验脚本；默认 DRIVE。 |

## 论文与实现对应关系

- 绿色通道/CLAHE 与三通道输入：`utils/dataloader_retina.py`。
- ResNet-34 五级特征：`lib/resnet.py`，使用标准卷积。
- AS-MSCB 与 SKU：`lib/decoders.py` 的 `AsymmetricDWConv`、`MSDC`、`MSCB`。
- RMSAB/GRAB：`lib/decoders.py` 的瓶颈路径与 `use_grab` 开关。
- AAF：`AdvancedAttentionFusion`。
- DBE：`lib/networks.py` 中的 `edge_extractor`、`edge_lgag`、`edge_fusion_head`。
- Focal + Dice + clDice 及深监督：`train.py`。

## 已在本审查稿中统一的内容

1. 以 `xiaorongshiyan` 为代码基线，排除数据集、日志、权重、预测图和实验表格。
2. 移除论文最终没有使用的 RSF adapter、四方向 Sobel 权重图及其文件、导入和命令行开关。
3. 默认设置与论文一致：ResNet-34、512px 输入、AdamW、初始学习率 1e-3、权重衰减 1e-4、batch size 4、200 epochs、10 个 warm-up epochs、最低学习率 1e-6、AMP、余弦退火、DRIVE/CHASE_DB1。
4. 依赖文件已改为可移植的公开包声明；旧环境快照仅保存在本地审查目录，不进入发布稿。

## 隐私与发布检查

已扫描项目源码和文档中的 API key、token、密码、Authorization、GitHub PAT 等常见敏感字段；未发现凭据。数据集、模型权重、日志和结果表均未纳入项目。

## 运行方式

`train.py` 按论文图像 ID 建立配对，不再独立排序后按位置配对。数据集根目录下需要有 `images`、`1st_manual`，以及 DRIVE 的官方 `masks`；推理输入会补齐到 32 的倍数，并按原始尺寸计算指标。`test_final.py` 默认在 validation 上校准阈值并启用 TTA，再评估未参与校准的 test split；消融实验使用 `--no-tta --fixed-threshold`。

## 当前状态

这是给作者检查的本地构建稿，尚未创建 GitHub 仓库、提交远程分支或发布归档。

## 验证记录

- `python -m compileall -q beamnet`：通过。
- CPU 前向测试通过，五个预测头均输出与输入相同的空间尺寸；解码器对 `timm` 只作可选兼容导入。
- 当前工作环境缺少 `albumentations` 与 `medpy`，因此使用了项目内的轻量数据加载和评估实现。

