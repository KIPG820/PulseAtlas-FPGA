# PulseAtlas 电极片识别与位置核对：委托实施计划

交接基线：2026-10-10 R12；本文件与“已有视觉成果”文件夹一起交付。**现阶段只有人体姿态与手工目标点标定；没有已训练的电极片识别模型，也没有完成“贴对”判断。**

## 1. 要实现的功能

PYNQ-Z2 的 USB 摄像头持续提供图像。在现有人体姿态模型投影出的目标位置上，新增自训练的电极片检测模型，找出画面中实际贴片的中心/边界。对同一摄像头帧进行一对一匹配，逐点输出：位置合适、需要调整、无法判断；HDMI 和 TFT 的视觉页显示目标圈、实际贴片框、偏差方向与汇总。保留原有 ECG 采集、AI 分析、触摸和 SW0 单/双屏功能。

视觉只能核对**看得见的几何位置**，不能证明皮肤接触、导线接通或 ECG 信号质量。贴片被遮挡、姿态不可信、标签身份不明时必须输出“无法判断”，不能将“没检测到”当成“贴错”或“贴对”。

**关键范围边界：**现有 RA、LA、RL、LL 四个绿色点是用户先前在上半身视角手工标定的躯干参考点，不是手腕/脚踝的四肢标准位置。用户后来说明实际使用标准四肢位置，因此旧四点不能作为四肢贴片的正确性真值。首期建议先核对画面内 V1–V6；标准四肢要增加覆盖四肢的视角和新标定，画面外一律“无法判断”。胸前参考位置也必须由懂 12 导联放置的人确认，不可只靠影像猜肋间隙。可参考 [AHA 的 12 导联位置示意](https://www.heart.org/-/media/Files/Affiliates/WSA/Montana/Mission-Lifeline-Montana/MLL-12-lead-Physio-ucm_484861.pdf?sc_lang=en)。

## 2. 已有成果（交接包内附原件）

| 文件 | 用途 |
| --- | --- |
| vision_guide/models/yolo11n-pose-cv454.onnx、model_manifest.json | 已部署的人体姿态模型，固定 256×192 RGB FP32 输入，在 PYNQ PS 上用 OpenCV DNN 运行。不是贴片识别模型。 |
| vision_guide/ps_onnx_pose.py | 原始未镜像 BGR 摄像头帧中选择完整躯干，输出 COCO 17 关键点与框；6/5/11/12 分别为右肩/左肩/左髋/右髋。 |
| vision_guide/electrode_placement.py | TorsoLeadCalibration 和 ChestLeadCalibration 把标定点从肩髋四边形投影到当前原图；V3 为 V2、V4 中点；TemporalPointStabilizer 抑制抖动。 |
| vision_guide/calibration/torso_lead_calibration.json、chest_lead_calibration.json | 用户之前手工标定的 v2 归一化锚点；不是通用解剖真值。人物、机位、姿态范围变化后重新核对。 |
| ps_vision_runtime.py | 板端摄像头采集、人体姿态推理、标定点投影和视觉页状态。新模型和融合逻辑建议在这里以独立模块接入。 |
| pulseatlas_r9_app.py、README_R12.md | 已验证 R12 的 UI/运行说明。SW0=0 仅 TFT、SW0=1 HDMI+TFT。 |
| vision_guide/test_electrode_placement.py、build_tools/test_ps_vision_runtime.py | 现有几何和线程运行测试，修改后要继续通过。 |
| vision_guide/README.md、run_webcam.py | PC 相机预览与重新手工标定步骤。PC 曾使用人体分割辅助标定；R12 板端只运行人体姿态模型。 |

板端运行实际用 R12 ZIP 里的上述文件。本交接包是从 R12 ZIP 提取的对应版本，标定 JSON、模型、几何代码已与 D 盘源码核对一致。完整 R12 部署包仍在 D:\ECG-fpga\artifacts\PulseAtlas_R9_HDMI_TFT_SW0_20261010_r12.zip，本交接包不包含 .bit/.hwh，也不修改板子。

**坐标合同：**所有人体点、目标点、贴片检测框中心都须换算回同一帧的**原始未镜像摄像头像素坐标**（通常 640×480）后才能比较。网络 256×192 坐标、HDMI/TFT 画面坐标、镜像自拍坐标不能直接相减；显示前才镜像。输出带 frame_id、采集时间、原图大小、置信度和模型版本。当前采集与推理是分线程的，显示帧可能比推理结果新，禁止将异时刻结果直接配对。

建议融合接口（伪代码，尚未存在）：

    pose = pose_model.detect(frame_bgr)
    targets = chest_calibration.positions(pose) if pose is not None else {}
    pads = pad_model.detect(frame_bgr)  # 原图坐标 (cx, cy, bbox, confidence)
    result = matcher.update(frame_id, captured_at, pose, targets, pads)
    # 每个目标返回 correct/adjust/unknown、距离、阈值、匹配分数、原因

## 3. 施工顺序

### A. 数据与标签

1. 固定实际 USB 摄像头、安装高度/距离、画面方向、光照和贴片款式；记录这些条件。采集不同人、肤色、体型、姿态、距离、亮度，以及正确、偏移、遮挡、反光、只有扣头/导线、无贴片、多人和画面外场景。
2. 第一版只标单类 electrode_pad 的可见边界框；“位置正确”不是外观类别，而由目标点和检测点的几何关系产生。把与贴片相似的扣头、皮肤圆斑、反光列为难负样本。约 1000–2000 张去重标注帧和至少 200 张独立盲测帧可作为启动目标，数量不是精度保证。
3. 按**人和采集会话**分训练/验证/盲测；同一段视频的相邻帧不可分到不同集合。标注规范写明部分遮挡和出画如何标注，双人复核边界样本。人体画面需取得授权、去标识化并受控保存。
4. 若所有贴片外观相同，模型只能识别“这里有一片”，无法凭外观证明它是 RA 还是 V2。若需要导联身份正确性，需可见编号/颜色或可靠导线追踪；否则匹配后仍提示“导联身份待人工确认”。

### B. 模型训练与板端导出

1. 在 PC/GPU 上训练轻量单类检测器，保存数据版本、训练配置、随机种子、best.pt、精度和误例图。官方 [Ultralytics 训练/验证 API](https://docs.ultralytics.com/usage/python) 和 [ONNX 导出说明](https://docs.ultralytics.com/modes/export)可供实施参考；选型按**独立盲测漏检/误检 + 板端耗时**，不只看训练 mAP。
2. 导出固定输入、静态形状 ONNX。在 PYNQ 实机先执行 OpenCV 4.5.4 的 readNetFromONNX 和单帧推理。现有人体 ONNX 抽取原始输出头，是为绕开该版 OpenCV 无法处理的动态解码尾；新模型导出成功并不代表能在板上跑。若转换图结构，交转换脚本和前后同图数值对照。
3. 分别试输入分辨率、躯干 ROI、量化 INT8，记录各自精度损失和 PS 耗时。双模型负载过高时先试低频姿态定位 + ROI 贴片检测 + 有时效上限的结果复用；不能为画面流畅展示过期的“贴对”。首版先跑通 PS，PL 加速列为性能优化阶段，不能替代正确性验收。

### C. 匹配、判定和 UI

1. 对同一帧的目标点与贴片中心做一对一全局匹配（匈牙利算法或少量目标的穷举）；一片不可配两个目标。匹配代价至少包括距离、检测置信度和目标区域。用肩宽/躯干尺度归一化距离，再在独立标注集上定阈值。阈值是相机/标定条件下的工程容差，不是医学位置标准。
2. correct 只表示“视觉上落入已校验的容差”，adjust 要求目标和贴片都清晰且偏移超阈值，unknown 覆盖低姿态置信度、遮挡、目标出画、多人、贴片身份不明、异帧、结果超时。稳定显示可要求连续若干次一致并加回滞，但记录原始逐帧结论。
3. HDMI/TFT 画目标圈、检测框、偏差方向和逐点状态；汇总“已核对 x/y、需调整、无法判断”。推理不得阻塞 ECG 采集、触摸或 SW0 切换。

### D. 验证与交付

1. 盲测报告：贴片 precision/recall、每帧误检、漏检、中心误差（像素与肩宽归一化）、correct/adjust/unknown 混淆矩阵；按光照、距离、遮挡、偏移幅度分层。建议在收集盲测集**之前**双方约定首轮指标，例如可见贴片召回 ≥95%、误检 ≤5%/帧、明确错贴场景 adjust 召回 ≥90%；这些是工程目标，并非当前已达结果。
2. 板端实测至少 10 分钟：ECG 丢帧和错误计数、CPU/内存、摄像头 FPS、人体模型 FPS、电极模型 FPS、融合输出 FPS、端到端延迟；验证 SW0 双屏、TFT 触摸、AI 页、摄像头拔插、无人/多人/遮挡、启停不死机。PC 与 PYNQ 在同一录制帧上比对坐标和输出。
3. 接手者必须交：数据与授权记录、标注规范、数据划分清单、训练/验证/导出脚本与锁定环境、best.pt/板端 ONNX/校验值、盲测报告及误例、融合源代码与回放测试、部署及回滚说明、板端日志和演示视频。不能只交权重或截图。若继续使用 Ultralytics 代码/权重，发布和移交前核查其[官方许可说明](https://docs.ultralytics.com/help/contributing)。

## 4. 里程碑

| 阶段 | 应看到的可检查结果 | 通过门槛 |
| --- | --- | --- |
| 0 基线复现 | PC 标定预览、PYNQ R12 视觉页、模型/JSON 版本与耗时 | 镜像、10 个目标点、SW0 行为和本包一致 |
| 1 贴片识别 | 独立盲测集、模型、误检/漏检报告 | 没有同人同段泄漏；板端真实载入并推理 |
| 2 位置融合 | 同帧一对一匹配、三态输出、录制帧回放 | 镜像、缩放、遮挡、多人、错贴不会产生假 correct |
| 3 板端集成 | 新部署包、ECG 完整性/FPS/延迟日志、回滚包 | 原有 ECG/触摸/SW0 可用，性能和误差达到约定值 |

姿态四边形无效、旧标定不适用、四肢贴片不在画面、贴片身份不明或新模型导致 ECG 丢帧时，显示“无法判断/需重新标定”，并回退到已验证 R12。

## 5. 本包快速复现

在工程根目录运行以下现有离线测试（依赖见 vision_guide/requirements.txt）：

    python ui/pulseatlas/vision_guide/test_electrode_placement.py
    python ui/pulseatlas/vision_guide/test_ps_onnx_pose.py
    python ui/pulseatlas/board_r9/build_tools/test_ps_vision_runtime.py

PC 相机预览和重新标定请读 vision_guide/README.md。交接包没有任何真人照片；实际拍摄数据需由用户另行授权并通过受控渠道提供。
