# PulseAtlas A 板 R12：界面与真实历史记录

R12 基于 A 板 R11，保留原有位流、ECG 采集、滤波 IP、推理模型和 DMA 配置；本次更新替换板端界面库和 Python 页面逻辑。

## 界面

- AI 页面只显示所选推理后端、分类分数和耗时；隐藏运行次数、PS/FPGA 一致性提示及页面底部的免责声明。为计算均值，所选后端仍对同一输入运行三次。
- 波形标注明确使用 `ADC 码` 和 `25 mm/s`。当前采集值没有校准成毫伏，因此界面不会把它标成 mV。
- TFT 实时监测、12 导联和单导联放大页面保留滤波开关与波形增益控制；滤波开关实际切换 PL 滤波器旁路，不重启 ADC/DMA。重新启用滤波会复位滤波器状态并清空分类器窗口，AI 开始前也会恢复标准滤波。
- PSD 单位移到频谱图上方，不与频率刻度或颜色条重叠。

## 历史记录

历史页从本机 SQLite 数据库读取真实采集会话，不生成示例记录。数据库默认位置为：

```text
/home/xilinx/.local/share/pulseatlas/history.sqlite3
```

可用 `PULSEATLAS_HISTORY_DB` 环境变量指定其他位置。采集线程把真实导联样本、帧号、质量标记和 R-R 间期送入有界队列，由单独后台线程压缩并写盘；推理完成后写入后端、分类、分数、概率和耗时。页面显示波形预览、采集时长、质量异常和实际推理结果。若采集块丢失或程序异常退出，该条记录标记为不完整。

为限制 SD 卡占用，默认最多保留最近 30 条会话，每条会话滚动保留最近 60 秒的波形；心率、采样总数和质量异常统计覆盖完整会话。可通过 `PULSEATLAS_HISTORY_MAX_SESSIONS` 与 `PULSEATLAS_HISTORY_MAX_SESSION_SECONDS` 调整保留范围。

## HDMI 声音

R12 没有增加 HDMI 音频。当前配套 HWH 中 HDMI 输出链使用 `rgb2dvi`，只有 TMDS 视频通路，没有 HDMI 音频数据岛或音频发送器；软件无法在这条现有位流上启用声音。要让 HDMI 有声，需要在 Vivado 设计中增加兼容的 HDMI Tx 音频通路并重新生成、验证匹配的位流。PYNQ-Z2 板载音频编解码器的耳机输出是独立模拟音频口，不等同于 HDMI 声音。

## 安装

将 R12 ZIP 解压到新的目录，先校验哈希，再运行：

```sh
sha256sum -c SHA256SUMS.txt
sudo -E /usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py
```

本地验证包括 Python 语法检查、真实历史数据库端到端测试、LVGL 页面渲染检查和 ARM Cortex-A9 交叉编译。未在板端验证的项目以实际板端回归结果为准。
