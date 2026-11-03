# B 板视觉加速：论文与开源实现审查（2026-10-12）

## 结论

下一轮应从当前 96 层 YOLO11n-pose 的逐层 PS/PL 调用，转向面向本任务重新训练的低比特、单次调用、流式关键点网络。架构依据优先采用 PYNQ-Z2 上 LPYOLO/FINN 的已公开部署流程；数据通路参考 SATAY 的流式层间连接、卷积行缓存和必要时 DDR 外部 FIFO。模型质量门槛沿用相机域人工标注集，旧的 COCO-only 小模型不进入部署。

## 公开案例核验

| 案例 | 平台和结果 | 可以借鉴 | 局限/可信度 |
|---|---|---|---|
| LPYOLO + FINN（论文、作者代码和部署包） | PYNQ-Z2 / XC7Z020；4-bit 重新设计 TinyYOLOv3 人脸检测；报告 18 FPS、2.4 W。公开仓库提供 QAT 导出、FINN 构建/部署流程及多种 W/A 位宽模型。 | 低比特 QAT、针对 XC7Z020 的 PE/SIMD folding、权重常驻片上、FINN 生成模型专用数据流、PS 仅处理前后处理和通信。这是最贴近 B 板的可复现先例。 | 人脸检测不能直接代替人体关键点；18 FPS 是其模型和任务的结果，不等同本项目速度。 |
| FPGA-SpacePose（论文和作者代码） | FINN + MobileNetV2/URSONet 混合精度关键点姿态网络；论文实测平台为 ZCU104，报告 58.7 FPS。 | 与本任务最相关的模型方向：紧凑 MobileNetV2 骨干、专用姿态输出头、逐层评估量化敏感度、FINN FIFO/折叠和整网数据流。源码包含训练、量化、构建及部署流程。 | ZCU104 资源远多于 XC7Z020，58.7 FPS 不可搬用；作者还报告 BRAM/FIFO 不足会令实测吞吐比估算低 4 倍以上，说明要以本板布线和实测为准。 |
| FINN 官方示例 | 官方工具链支持生成 AXI-Stream 数据流加速器，并列出 PYNQ-Z2 支持；但多数现成 PYNQ 示例面向 PYNQ-Z1，Z2 目标需调整构建配置。 | 复用 FINN/QONNX 图转换、折叠配置、自动 FIFO sizing、PYNQ driver/deployment package 流程。 | 通用 YOLO11-pose 并非即插即用；算子和输出头需改成 FINN 可支持的图。 |
| SATAY（YOLO 流水数据流） | 全图流式、模型参数片上；卷积使用行缓存，残差/跳接缓冲按资源需要放片上或 DDR。论文测试平台为较大 FPGA。 | 中间张量不逐层回 PS；小卷积使用 `(K−1)×W×C` 级别行缓存；只在跨层依赖和容量确实需要时落 DDR。 | 论文平台不是 XC7Z020，不能把其 FPS/资源结论平移到 B 板。 |
| YOLOv11 非法采矿 FPGA 论文 | 文章声称 PYNQ-Z1 上 YOLOv11 18 FPS/55 ms，但架构描述依赖 DDR 中的权重和特征图、PS 端 NMS。 | 可参考 PS/PL 划分、DMA 和像素缓冲的系统结构。 | 文章没有给出可复现的完整算子映射/量化流程；还报告 8.3 Mbit 片上存储，而 XC7Z020 标称 BRAM 约 4.9 Mbit，资源数字存在明显矛盾，因此不把 18 FPS 当作可移植证据。 |
| ICTA 2025 超轻量姿态论文 | 摘要报告 0.4M 参数、ConvMixer/FPN、FPGA 21.5 FPS。 | 证明关键点任务可以从专用小模型与合适加速器受益，而不是必须完整运行通用 YOLO pose。 | 公开摘要未明确 FPGA 型号、端到端计时边界或源码；只作为模型方向线索，不作为本板性能承诺。 |
| 2026 FINN 车辆检测预印本 | PYNQ-Z1/XC7Z020；LP-YOLO Slim、256×256、W2A4、142.86 MHz；给出 35.66 FPS（batch=100）、45.11 ms PL latency、49.77 ms end-to-end latency、VOC mAP@0.5=0.594。 | 同器件上的低比特 QAT、架构瘦身、folding/FIFO 调优和板端实测流程。 | 这是车辆检测预印本；batch 吞吐和单帧响应时延是不同指标，不能把 35.66 FPS 当作当前单帧关键点模型的实测值。 |

## 与 B 板实测的关系

R3 的历史摄像头端到端结果约 0.9–1.0 次推理/秒。2026-10-12 在 B 上做了无逐层跟踪的热态完整图复测：R3 平均 0.737 秒/图（1.36 图/秒）；其卷积后端约 0.526 秒，其中 CPU im2col 打包约 0.247 秒、PL 卷积约 0.162 秒。早期 1.12 秒/图的数据在计时区间内逐层扫描黄金特征图，包含验证开销，不应当作生产图耗时。

R4 raw-im2col 候选在同板、同一输入和同一量化图下逐元素通过 96 个卷积输出与 3 个最终输出的黄金回归。分离验证与计时后，R4 五次无跟踪热态平均 0.709 秒/图（1.41 图/秒），相对 R3 仅降低 3.8% 延迟。它把约 0.247 秒的 CPU 打包移到 PL，但 HLS 卷积时间从 0.162 秒增至 0.276 秒，总卷积后端时间只从 0.526 秒降至 0.496 秒。B 服务已恢复运行，但本次没有连接摄像头，因此这不是端到端 FPS。仅把整块 im2col 挪进当前单层 PL 核没有带来突破；关键仍是减少 96 次启动和中间张量往返，并降低模型运算量。

现有 224×160 试验只有约 8% 软件推理缩短，定位精度明显变差，故不把单纯降分辨率作为主路径。现有 20k–100k 参数四点学生在 COCO-only 留出集上的 PCK@0.10 仅约 0.15–0.21，也未通过质量门槛；它说明当前训练数据/模型试验不足，不证明小模型不可行。

## 2026-10-12 实现复核与决策

作者的 LPYOLO 仓库确认了可复用的实现流程：Brevitas QAT 训练量化网络，导出 FINN 兼容 ONNX，FINN 生成定制数据流 IP 和 PYNQ 驱动；输入/输出可保持 INT8，权重/激活可用更低位宽，部署包还包含 PYNQ-Z2 位流。作者仓库列出的模型包括 2W4A、4W4A 等；论文案例将整个网络参数保存在片上，并由 PS 承担预处理、后处理和网络通信。这里可复用的是量化训练、算子约束、硬件折叠和端到端部署流程，不能直接拿人脸检测模型替代本项目的身体关键点检测。

对当前 R3 的 INT8 ONNX 图做了算子盘点：333 个节点中有 96 个 QLinearConv、84 个 QLinearSigmoid、84 个 QLinearMul、20 个 QLinearConcat、14 个 QLinearAdd，以及 Resize、Split、Reshape、MatMul 等节点。仅将 96 个卷积逐层改由 PL 执行，仍保留大量 PS 节点与逐层数据搬运，和 LPYOLO/FINN 的全图数据流架构不同，因此不能据此预期达到 5 FPS。

当前 v12 raw-im2col HLS IP 是一个对照基线：C 仿真和 RTL co-sim 已通过，估算频率 136.99 MHz。隔离 PS/DDR/3×HP 系统已完成 Vivado 2024.2 综合、路由和 bitstream 生成；路由 WNS 为 +0.104 ns、无路由错误，资源为 22,061 LUT、222 个 RAMB18 等效块（约 79%）、156 DSP（约 71%）。板端完整整数图正确，但每个卷积仍单独启动并在 DDR/PS 间传张量；R4 的板端数据说明此核不能作为最终架构结论。板端记录和服务状态见 `docs/measurements/20261012/r4_raw_im2col_board_validation/`。

后续主线采用 FINN/LPYOLO 的实现原则：将模型改成少量、硬件友好的低比特卷积算子；通过针对任务的 QAT 保住关键点精度；用 folding 分配 PE/SIMD 并把连续层接成片内流；只在输入、输出及必要的长跳接处访问 DDR。当前姿态 ONNX 不适合直接套 FINN：需要验证是否能将激活/Resize/Concat/检测头改写成可综合形式，若代价过高，则以现有姿态模型产生候选标注，再训练一个四点关键点专用轻量网络。该模型必须先在相机域留出集达到定位精度门槛，再进入硬件实现。

模型架构优先参考 FPGA-SpacePose 的 MobileNetV2 + 专用姿态头与逐层混合精度评估流程，再为四个身体定位点删减通道和输出头；硬件生成与数据流组织参考 FINN/LPYOLO。若四点定位任务现有标注不足，先从相机域整理有人工核验的训练/留出样本；不能仅凭 COCO 关键点或合成样本宣称定位质量合格。

## 执行路线

1. **任务化模型**：以四个贴片引导关键点为输出，采用紧凑的普通卷积/ConvMixer-FPN 结构；避免直接照搬为 FPGA 不友好的通用 YOLO11 图。优先借鉴 LPYOLO 的 QAT 与 FINN 支持层集合，并参考姿态论文的轻量 FPN/ConvMixer，而不复制未经验证的层宽。
2. **量化**：先做 W4A4 和 W2A4 量化感知训练候选；输入和输出保持 8-bit，首层/末层按验证需要提升位宽。逐层评估量化误差，选择满足定位精度的最低位宽。
3. **硬件图**：一次 AXI-DMA 输入帧、一次启动、一次输出关键点；卷积/激活融合，层间 AXI-Stream，卷积行缓存；用 FINN folding 和 FIFO sizing 适配 XC7Z020。权重尽可能常驻 BRAM，超过容量时再评估顺序权重流，不预设整网权重必然能放入片上。
4. **部署边界**：PS 保留 USB 采集、裁剪/缩放、界面和 TCP 预览；PL 负责紧凑网络的主要计算；输出少量坐标/置信度。以同一计时脚本分别记录采集、预处理、PL、后处理、预览和完整帧率。
5. **放行门槛**：先在相机域留出集上达到当前模型的可接受误差和无人体误报要求，再做 QAT 与 RTL 对照；HLS C 仿真、RTL co-sim、综合/布线时序通过后才生成独立 B-only 位流；板端连续运行无内存增长，完整摄像头到关键点达到至少 5 FPS 后才替换 R3。

## 当前环境约束

2026-10-12 本机检查确认 Vitis HLS/Vivado 2024.2、PyTorch、ONNX 和 Ultralytics 可用；Brevitas、FINN、Docker 及已安装的 WSL Linux 发行版不可用。Vitis 安装位于 `D:/Xilinx/`，当前 `XILINX_VITIS` 环境变量却指向不存在的 `D:/xinlix/`，后续 HLS 构建需在当前进程修正工具路径。FINN 自动系统集成和 CI 不再覆盖 PYNQ-Z1/Z2，需要手动维护 XC7Z020 构建。B 板当前 SSH 可达，R3 服务已恢复；本次测试后 `/dev/video*` 缺失，因此需摄像头重新连接后再做实时 FPS 与画质复核。

## 参考资料

- Bestami Günay et al., *LPYOLO: Low Precision YOLO for Face Detection on FPGA*, PYNQ-Z2, 2022: https://arxiv.org/abs/2207.10482
- 作者代码与部署包： https://github.com/sefaburakokcu/finn-quantized-yolo
- Julien Posso et al., *Real-Time Spacecraft Pose Estimation Using Mixed-Precision Quantized Neural Network on COTS Reconfigurable MPSoC*, 2024: https://arxiv.org/abs/2407.06170
- 作者实现（MobileNetV2 + URSONet + FINN）： https://github.com/possoj/FPGA-SpacePose
- FINN 官方数据流示例： https://github.com/Xilinx/finn-examples
- FINN PYNQ-Z2 构建/部署说明（作者）： https://medium.com/@bestamigunay1/end-to-end-deployment-of-lpyolo-low-precision-yolo-for-face-detection-on-fpga-13c3284ed14b
- SATAY, *A Streaming Architecture Toolflow for Accelerating YOLO Models on FPGA Devices*: https://arxiv.org/abs/2309.01587
- Ghodhbani et al., *Hardware-Accelerated Detection of Unauthorized Mining Activities Using YOLOv11 and FPGA*, IJACSA 16(4), 2025: https://doi.org/10.14569/IJACSA.2025.0160494
- Gutierrez et al., *NPU Accelerator: Quantized Real-Time Vehicle Detection on PYNQ-Z1 Using FINN*, arXiv:2609.24757, 2026 preprint: https://arxiv.org/abs/2609.24757
- Cao et al., *Design and Optimization of a 0.4M-Param. Ultra-Lightweight Human Pose Estimation Network Based on Edge CNN Accelerator*, ICTA 2025 session listing/abstract: https://epapers2.org/icta2025/ESR/paper_details.php?paper_id=9258
- AMD, Zynq-7000 SoC Family Product Selection Guide (XC7Z020 4.9 Mbit BRAM): https://docs.amd.com/api/khub/documents/1L_hkh2pbc5l0Oz7tcLspA/content
