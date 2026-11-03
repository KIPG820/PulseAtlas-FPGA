# ads1298_acquire_ip

PYNQ-Z2 / ADS1298 采集 IP。目标 PL 时钟为 100 MHz，适配前端原理图中的单片 ADS1298IPAGR。该器件有 **8 个同步采样通道，每通道 24 位**；本 IP 只支持 500/1000 SPS 下的 216 位帧（24 位状态字 + 8×24 位数据），不支持高速模式下的 16 位帧。

## 端口与数据格式

- `ecg_mosi`→ADS1298 DIN；`ecg_miso`←DOUT；`ecg_sclk`→SCLK；`ecg_cs_n`→CS#；`ecg_ready_n`←DRDY#；`ecg_start`→START；`ecg_reset_n`→RESET#；`ecg_pwdn_n`→PWDN#。后两者是独立信号。
- `m_axis_ecg8_tdata[255:0]`：CH1 在 `[31:0]`，CH8 在 `[255:224]`，每路 24 位二补码符号扩展至 32 位。
- `m_axis_ecg8_tuser[31:0]`：`frame_seq`；`[55:32]`：ADC 状态字；其他位清零。每个 ADC 采样帧传输一次，`TLAST=1`。
- `ecg_ready_n` 经三级触发器同步。ADS1298 自行采样，不接受下游背压；内部两帧 FIFO 满时丢帧，序号仍递增，并记录溢出。

## 初始化和工作方式

1. 保持 PWDN# 与 RESET# 为低，等待电源稳定；释放 PWDN# 与 RESET# 后默认再等待 200 ms，使内部振荡器与电源复位完成，再单独对 RESET# 施加低脉冲并释放。
2. 发送 `SDATAC`，读取 ID 并要求等于 `0x92`（ADS1298，8 通道）。
3. 写入并逐个读回 CONFIG1/2/3、LOFF、CH1SET～CH8SET、RLD、LOFF_SENS、WCT1/2；失败时进入错误态，不启动采样。
4. 等待内部参考电压稳定。默认进入空闲检测态，保持 ADC 上电但停在 `SDATAC`，每秒轮询一次 ID；CTRL[5]=1 时发送 `RDATAC`、拉高 START 并按 DRDY# 读取完整数据帧。CTRL[5] 清零会发 `SDATAC` 并回到检测态，不会复位或重新初始化 ADC。

设备未识别时每 250 ms 重试完整初始化；连续采样期间若 500 ms 没有 DRDY#，认为器件掉线并进入重试流程。ID 轮询失败会使 `DEVICE_ID` 与 STATUS[7] 显示无效，但不增加寄存器配置错误计数。

默认配置为高分辨率 500 SPS、8 路电极输入、增益 6、内部 2.4 V 参考、RLD 取 LL/RA/LA、WCT 取 RA/LA/LL。前端 CLKSEL 已接 3.3 V，因此使用 ADS1298 内部主时钟。数字电源 3.3 V。SPI 模式为 CPOL=0、CPHA=1，默认 SCLK 为 2 MHz。采用时钟使能产生 SCLK，PL 状态机始终使用 `aclk`，不把 SPI 时钟当作内部逻辑时钟。

`CTRL[2]=1` 时，CONFIG2 使用内部测试源且 CHnSET 选择测试信号。写入配置后，需在 `CTRL[1]` 写 1 触发重新初始化；采样率和增益也在重新初始化时生效。`CTRL[3]=1` 选择 1000 SPS，默认 500 SPS。

## AXI4-Lite 寄存器（字节偏移）

| 偏移 | 名称 | 含义 |
|---|---|---|
| `0x00` | CTRL | bit0 使能；bit1 写 1 重新初始化（读回为 0）；bit2 内部测试信号；bit3 1000 SPS；bit5 连续采样/数据流使能；bit4 保留。bit5=0 时保持上电并轮询 ID；复位值 `0x01`。 |
| `0x04` | STATUS | bit0 使能；bit1 错误态；bit2 配置错误；bit3 帧头错误；bit4 溢出；bit5 连续采样运行中；bit6 SPI 读帧中；bit7 最近一次 ID 检测有效。写 bit0=1 清除粘滞标志及统计计数。 |
| `0x08` | FRAME_SEQ | 下次 DRDY# 对应的帧序号。 |
| `0x0C` | FRAME_COUNT | 成功放入内部 FIFO 的帧数。 |
| `0x10` | DROP_COUNT | 丢帧数。 |
| `0x14` | FRAME_ERROR_COUNT | 状态字前导不等于 `1100` 的次数。 |
| `0x18` | CONFIG_ERROR_COUNT | 配置寄存器回读错误次数（ID 读取失败由 STATUS[7] 和 DEVICE_ID 表示，不计入此项）。 |
| `0x1C` | SPI_HALF_PERIOD | 半周期为 `(此值+1)` 个 `aclk` 周期；默认 24，即 2 MHz。允许写入值 ≥4。 |
| `0x20` | CH_GAIN | CHnSET 的 `[6:4]`，8 路相同；默认 0 对应增益 6。 |
| `0x24` | LOFF | ADS1298 LOFF 寄存器，默认关闭。 |
| `0x28` | LOFF_SENSP | 正端检测掩码，默认 0。 |
| `0x2C` | LOFF_SENSN | 负端检测掩码，默认 0。 |
| `0x30` | DEVICE_ID | 最近一次 ID 读取值。 |

若启用导联脱落检测，须先选择有效 LOFF 频率（例如 DC 检测对应 LOFF[1:0]=`11`），再设置 SENS 掩码，之后重新初始化。不要把此功能和接人体电极的安全验证混为一谈。

## 验证及接入

在 PowerShell 中运行 `sim/run_xsim.ps1`，脚本使用本机 Vivado 2024.2 的 `xvlog/xelab/xsim`，测试 SPI 命令、寄存器回读、通道顺序和符号、AXI 背压、FIFO 溢出及重新初始化。仿真是模型验证，尚非实板验证。

RTL 文件位于 `rtl/`。实物插接时 H1 的偶数列对应 PmodA 左列，故 H1-2（MOSI）位于 JA1P；H1-9/10 为 GND，H1-11/12 由 PmodA 供应 3.3 V。八根信号线的独立约束见 `constraints/pmoda_ads1298.xdc`。本 IP 已接入 `ecg_system`，原 PmodA IOP 已移除，采样流经 AXI DMA 写入 DDR，HDMI 链路保留；生成文件和上板测试 Notebook 位于 `PYNQ-3.1.1/boards/Pynq-Z2/ecg_system/deploy/ecg_ads1298/`。实板测试仍需核对 SCLK/DRDY/DOUT 的实际时序与信号完整性。

依据：[TI ADS1298 数据手册](https://www.ti.com/lit/ds/symlink/ads1298.pdf)第 7.6、9.5、9.6、11.1 节；本项目 `docs/references/SCH_Schematic1_2026-09-14.pdf`。
