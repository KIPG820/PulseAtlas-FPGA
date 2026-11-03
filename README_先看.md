# PulseAtlas 双 PYNQ 完整精简源码包

生成日期：2026-11-03

## 目录
- `01_A_board_current_runtime`：A 板 R14 完整运行文件，含 bit/hwh、ECG 模型、HDMI/TFT 和 A-B 通信。
- `02_A_FPGA_and_ECG_NN_sources`：ECG 采集、滤波、QRS 等 RTL IP 及 INT8 五分类网络 HLS 源码。
- `03_B_board_vision_runtime`：R7 完整依赖基线，叠加 R16 位流、R19 运行优化、R20 电极可靠性和最终重训小模型。
- `04_training_and_PC_tools`：ECG 训练/量化代码和 PC 两阶段姿态+电极检测工具。
- `05_deployment`：启动和部署配置。
- `06_technical_docs_and_reports`：架构、接口、资源时序、测试与复现说明。
- `06_technical_docs_and_reports/12_最终提交材料`：最终设计报告 PDF 与项目海报 PPTX。

## 精简范围
未包含原始数据集、训练缓存、Vivado/HLS 自动生成目录、旧压缩包、日志、视频及缓存文件。以上内容不影响阅读源码、板端部署或从源码重新构建。

## 版本口径
A 板为 R14 完整发布文件并覆盖此前从实际板端读取的核心源码；B 板为 R16 hardware + R19 runtime + R20 electrode reliability。B 监听 TCP 8765，A 为客户端。
