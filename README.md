# PulseAtlas 双 PYNQ 心电与视觉辅助系统

PulseAtlas 是基于两块 PYNQ-Z2（XC7Z020）的异构计算原型。A 板负责 ADS1298 十二导联心电采集、PL 流式预处理、QRS/RR 计算、ECG 五分类推理和 HDMI/TFT 交互；B 板负责姿态定位、电极候选区域生成及视觉推理加速，两板通过以太网传递状态和视觉结果。

## 主要内容

- `01_A_board_current_runtime`：A 板当前运行文件、位流、模型和界面程序。
- `02_A_FPGA_and_ECG_NN_sources`：ECG 采集、质量检测、滤波、小波、QRS、导联重构、打包等 RTL，以及 ECG INT8 推理 HLS 源码。
- `03_B_board_vision_runtime`：B 板视觉运行版本、加速位流和电极可靠性逻辑。
- `04_training_and_PC_tools`：ECG 训练与量化代码、PC 两阶段姿态和电极检测工具。
- `05_deployment`：部署脚本和开机启动配置。
- `06_technical_docs_and_reports`：系统说明、接口、资源时序、测试材料、最终设计报告和项目海报。

## 复现入口

请先阅读 [`README_先看.md`](README_先看.md) 和 `06_technical_docs_and_reports/08_版本与复现/复现与部署说明.md`。文件完整性可通过 `SHA256SUMS.txt` 核验，`FILE_MANIFEST.json` 记录了文件大小与哈希值。

## 数据与版本说明

仓库未包含原始人体数据集、训练缓存、Vivado/HLS 自动生成目录、历史压缩包及录制视频。A 板口径为 R14；B 板由 R16 硬件、R19 运行优化及 R20 电极可靠性版本组成。模型输出仅用于原型展示和辅助研究，不作为临床诊断结论。

## 许可证

当前私有阶段尚未选择最终开源许可证。第三方模型和组件的声明见各目录中的 `THIRD_PARTY_NOTICES.md`。仓库公开前应完成许可证和数据授权复核。
