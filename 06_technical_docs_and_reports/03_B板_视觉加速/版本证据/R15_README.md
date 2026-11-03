# B 板视觉 R15：保留现有模型的选择性零拷贝

本版不使用教师模型、不蒸馏，也不改变 YOLO11n-pose 的 INT8 权重、96 层图、电极存在检测 ONNX、颜色模型或人体标定。PL 继续使用已通过时序和逐层验证的 R14 90.909 MHz 位流。只修改 B 板 `vision_int8_graph.py` 和 `vision_int8_prepared_pl.py` 的数据传递：对“卷积输出只被下一层 PL 卷积使用”的 47 层保留 PYNQ 连续缓冲区视图；凡是后续 PS 需要做 Concat、Add 等运算的输出，仍复制到普通 NumPy 内存。每层 PL 直通输出使用独立小缓冲区，避免跳连张量被覆盖。全部 96 层输出合计约 2.03 MiB，本版仅让其中一部分常驻连续内存。

2026-10-15 板端同输入对照：96 个卷积中间结果、3 个最终输出和变体输入指纹均完全一致；9 次计时平均整图耗时 R14 为 0.453070 s，R15 为 0.424104 s，提升 6.83%（仅整图，不含相机与电极检测）。记录见 `validation/board_r14.json` 与 `validation/board_r15.json`。不加选择地让所有层零拷贝的实验反而从 0.458 s 退化到 0.503 s；原因是 PS Concat 和 Add 读取连续缓冲区明显变慢，故正式版只采用选择性直通。

相同摄像头服务、相同位流、相同电极/颜色模型下，42 秒订阅测试在 20 秒预热后的稳态计数：R14 为 1.532 FPS，R15 两次分别为 1.611 和 1.578 FPS；R15 最近一帧模型耗时约 591–594 ms，无服务或电极模块错误。记录见 `validation/live_r14_baseline.json`、`validation/live_r15_trial.json` 和 `validation/live_r15_repeat.json`。此画面未检测到人体，电极模型确认已加载，但真人贴片判断还需另测。

已将 R15 设为 B 板开机自启服务。运行目录为 `/home/xilinx/PulseAtlas_B_Vision_R15_ZeroCopy`，含两个新 Python 文件；原 R7 模型、标定、其余运行文件通过链接复用，原模型 ONNX 通过硬链接复用以保持正确导入路径，不复制大型模型或构建产物。服务仍引用 R14 位流和原电极/颜色模型；旧 R14 服务配置备份在 `/etc/systemd/system/pulseatlas-b-vision.service.r14-backup-20261015`。板端验证 `systemctl is-active` 为 active、`is-enabled` 为 enabled，剩余可用内存约 296 MiB。

当前完整视觉推理仍远低于 5 FPS；本版是安全保留原模型的增量优化，不应将相机采集 FPS 或预览 FPS 当成神经网络推理 FPS。
