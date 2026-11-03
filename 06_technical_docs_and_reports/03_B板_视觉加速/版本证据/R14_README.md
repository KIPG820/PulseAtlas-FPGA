# B 板视觉 R14：R12 IP @ 90.909 MHz

保持 R12 的 256×192 INT8 模型、96 层卷积、寄存器 ABI 与电极检测软件；把 B 板 PS 产生的 FCLK_CLK0 从 100 MHz 调低并重新布局布线。Vivado 对请求的 95 MHz 生成的实际频率为 90.909 MHz。此目录引用 R12 已验证的 HLS IP，不复制大型构建中间文件。A 板保持原样。

R12 的 96 层真实卷积 C 仿真逐字一致、44 组 C/RTL 联合仿真通过；在 100 MHz 下布线 WNS=-0.263 ns，布线后优化=-0.153 ns。Vivado 将请求的 95 MHz 映射为 **90.909 MHz**（11 ns，FCLK0 除数 11）。本版正式布线 WNS=+0.127 ns、WHS=+0.016 ns，路由错误 0、DRC 错误 0；`.bit` 和匹配的 `.hwh` 已生成。HWH 保留 `vision_conv_0` 的 0x43C00000 控制地址、同一 IP VLNV 和 FCLK0 除数。PYNQ 3.1.1 的 `Overlay.download()` 会从 HWH 的 `clock_dict` 设置 PS 的 PL 时钟，因此板端测试必须使用该 `.bit/.hwh` 成对加载；仅裸写 bitstream 可能仍保持旧的 100 MHz 时钟。

2026-10-15 在 B 板完成同输入板端测试：96 个卷积中间结果、3 个最终输出逐字一致；9 次预热后计时的整图平均耗时，R7 为 0.536142 s，本版为 0.456947 s，提升 17.33%（2.1884 图/秒）。R7 的卷积核合计耗时 0.288335 s，本版为 0.208921 s。记录见 `validation/board_r7.json` 与 `validation/board_r14.json`。测试退出已恢复 R7 位流与原服务，B 板 `/dev/video0` 存在。摄像头当前没有拍到人，因此完整摄像头、电极检测和预览链路的推理 FPS 尚未测到；空场景采集约 3.73 FPS，不等同于模型推理 FPS。

当前整图除 PL 核外仍约耗时 0.248 s，因此即使把卷积核耗时降为零，现有图执行路径也达不到 5 图/秒。下一阶段须同时削减 PS 图算子、DDR/缓存搬运或减少图调用开销。`tools/run_board_comparison.sh` 可重跑公平对照，退出时恢复原位流与服务。打包脚本只包含板端临时测试所需文件。

2026-10-15 已将本版设为 B 板正式服务位流，文件放在 `/home/xilinx/PulseAtlas_B_Vision_R14_90MHz/vision_accel/`。`pulseatlas-b-vision.service` 仍用原 R7 模型运行目录、原电极存在检测 ONNX、原颜色模型和标定文件，只增加 `--bitstream` 指向本版成对 `.bit/.hwh`。原启动配置备份在 `/etc/systemd/system/pulseatlas-b-vision.service.r7-backup-20261015`。服务 enabled、active，TCP 8765 正常监听；加载后的 FCLK0 实读 90.909091 MHz。28 秒现场订阅计数为 28 次推理/19.062 秒稳态窗口，即 1.469 FPS，摄像头采集约 12.87 FPS，预览 130 包；详细记录见 `validation/live_r14_after_warmup.json`。电极检测状态显示模型已加载、无报错；该次画面没有人体，因此十个电极位置仍为 unknown，尚未检验真人贴片识别结果。
