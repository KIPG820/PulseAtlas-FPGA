# PYNQ-Z2 ECG 系统：PL IP 开发与接口规范

> 目的：供团队成员独立开发、仿真和交付各自的 PL IP。本文只规定采集、数字预处理、导联重建、节拍定位和输出接口；神经网络加速不在本轮范围内。

## 1. 已确认的硬件事实与边界

- 目标 FPGA：PYNQ-Z2 的 XC7Z020，Vivado 2024.2。集成基线为工作区中的 `PYNQ-3.1.1/boards/Pynq-Z2/ecg_system/ecg_system.xpr`。
- ECG 前端芯片：**ADS1298IPAGR**，不是 ADS1198。参考随文档保存的[前端原理图](references/SCH_Schematic1_2026-09-14.pdf)和 [TI ADS1298 数据手册](https://www.ti.com/lit/ds/symlink/ads1298.pdf)。
- 原理图通道：CH1＝II（LL−RA），CH2＝III（LL−LA），CH3＝V2，CH4＝V3，CH5＝V4，CH6＝V5，CH7＝V6，CH8＝V1。胸导联负输入接 ADS1298 的 WCT 输出。WCT 的 RA、LA、LL 来源需要在芯片寄存器中配置。
- 前端数字线：`ECG_MOSI`、`ECG_MISO`、`ECG_SCLK`、`ECG_CS`、`ECG_READY`（ADS1298 的 `DRDY#`）、`ECG_START`、`ECG_RESET`。具体 PmodA 针脚号**尚未给出**；任何人不得凭猜测编写最终 XDC。原理图上 RESET# 与 PWDN# 相连，采集 IP 的复位时序必须按此连接处理。
- 前端模拟输入已串联 22.1 kΩ、对 AGND 并联 47 pF，单极点标称截止频率约 153 kHz。数字算法不能恢复模拟链路已衰减且落入噪声的成分。
- 第一版目标采样率为 **500 SPS**；采集 IP 保留切换到 **1000 SPS** 的能力。两者均按 24 位状态字＋8 路各 24 位数据，即 **216 位/帧**解析。32 kSPS 等特殊模式的 16 位输出不在第一版范围内。
- 保留当前工程 HDMI 输出。接入自定义采集 IP 时，需要从 Block Design 中移除占用 PmodA 引脚的通用 PmodA IOP 及其引脚控制链路；不能让两个模块同时驱动 PmodA。

**必须保留两种数据**：未经数字滤波的原始 8 路数据，以及用于显示/后续分析的处理后数据。质量差的片段要标记，不能静默用平滑值替代原始数据。

## 2. 共用设计约定（所有负责人必须遵守）

### 2.1 时钟、复位与握手

- 全部自定义 IP 在同一个 PL 时钟域工作，目标时钟为 100 MHz；实际接入时核对 Block Design 的 `FCLK_CLK0` 频率。统一采用低有效 `aresetn`，释放复位须同步到本时钟域。
- `ECG_READY` 来自 ADC 时钟域，进入采集状态机前至少经过同步；不得把它直接当 PL 逻辑时钟。
- IP 间数据接口为 AXI4-Stream。只有 `TVALID && TREADY` 同时为 1 时才算传输。阻塞期间 `TDATA/TUSER/TLAST` 必须保持稳定。
- 内部一帧指 ADS1298 的一次 8 通道同步采样，而不是 SPI 的一个字节。内部 AXI4-Stream 每帧只传输一次，`TLAST=1`。输出打包 IP 再转换成面向 DMA 的数据块。
- 采集源不能反向暂停 ADS1298；其 FIFO 满时允许丢帧，但必须设置粘滞 `overflow` 标志、增加丢帧计数，且下一帧序号继续递增，让丢失可定位。

### 2.2 统一帧接口

为避免各组在字节顺序、符号和质量标记上各用一套格式，固定如下内部接口：

| 接口 | 定义 |
|---|---|
| `ecg8_axis` | `TDATA[255:0]`，8 个有符号 32 位槽位；CH1 在 `[31:0]`，依次到 CH8 `[255:224]`。原始 24 位 ADC 数值做符号扩展，不做单位换算。 |
| `ecg12_axis` | `TDATA[383:0]`，12 个有符号 32 位槽位；依次为 I、II、III、aVR、aVL、aVF、V1、V2、V3、V4、V5、V6。 |
| 两种接口共用的 `TUSER[127:0]` | `[31:0]` 帧序号；`[55:32]` ADS 状态字；`[63:56]` 8 路电极脱落掩码；`[71:64]` 饱和掩码；`[79:72]` 平线掩码；`[87:80]` 突变掩码；`[99:88]` 12 导联无效掩码；`[127:100]` 保留且置零。掩码位 1 表示异常。 |

`ads1298_acquire_ip` 生成连续帧序号和状态字，其他字段清零；`ecg_quality_ip` 填写质量掩码；`ecg_lead12_ip` 填写 12 导联无效掩码。其余 IP 原样传递未修改的 `TUSER`。所有输出都要保持与原始帧序号的一一对应关系；流水线延迟允许存在，但不得改变序号或帧顺序。

配置/计数使用 AXI4-Lite。每个 IP 至少提供 `CTRL`（使能、软复位或旁路）、`STATUS`（就绪、错误、溢出）、输入帧计数、输出帧计数、错误计数；算法参数由各负责人在本文件约束下增设并写入各自寄存器说明。更改系数时必须定义生效边界，第一版统一在下一帧开始生效。

### 2.3 定点与数据有效性

- 外部输出仍以 **ADC count** 为单位；滤波器的内部系数和累加器可采用更宽定点，输出统一饱和到有符号 32 位。禁止不受控的环绕溢出。
- 质量为坏的帧仍按原样传递采样值和掩码；下游可选择忽略它。滤波器在复位、启用或系数切换后的暂态，应在 `STATUS` 中报告，负责人需记录其持续帧数。
- 第一版不把算法滤波等同于临床诊断滤波。尤其基线处理要比较 ST 段形态偏差；QRS 专用强滤波结果不能替代原始/形态保真数据。[相关方法比较](https://pmc.ncbi.nlm.nih.gov/articles/PMC5361052/)

### 2.4 端口命名、方向和数据契约

本节是七个 IP 的**集成接口约定**。后文算法实现方式可调整，以下端口名称、方向、位宽和字段含义未经全组协商不得更改。

**全部 IP 的公共输入**：`aclk`（1 位，目标 100 MHz）、`aresetn`（1 位，低有效）、AXI4-Lite 写地址 `s_axi_awaddr[7:0]`、`s_axi_awvalid`、写数据 `s_axi_wdata[31:0]`、`s_axi_wstrb[3:0]`、`s_axi_wvalid`、写应答接收 `s_axi_bready`、读地址 `s_axi_araddr[7:0]`、`s_axi_arvalid`、读应答接收 `s_axi_rready`。

**全部 IP 的公共输出**：AXI4-Lite `s_axi_awready`、`s_axi_wready`、`s_axi_bresp[1:0]`、`s_axi_bvalid`、`s_axi_arready`、`s_axi_rdata[31:0]`、`s_axi_rresp[1:0]`、`s_axi_rvalid`。每个 IP 是独立 AXI4-Lite 从设备，地址空间互不重叠。A 组已实现的寄存器偏移见其 README；B～G 组须在交付物中固定各自寄存器表，不能仅写“参数可配置”。

**AXI4-Stream 方向规则**：`s_axis_X_tdata/tuser/tvalid/tlast` 是 IP 输入，`s_axis_X_tready` 是 IP 输出；`m_axis_X_tdata/tuser/tvalid/tlast` 是 IP 输出，`m_axis_X_tready` 是 IP 输入。QRS 事件接口不带 `TUSER`，打包输出额外包含 `TKEEP[3:0]`。对采样帧和 QRS 事件，`TLAST` 恒为 1；对 DMA 记录流，仅在块的最后一个 32 位字置 1。任何下游背压期间，输出的 `TDATA/TUSER/TKEEP/TLAST` 必须保持不变。

| IP | 专用输入信号（除上述公共输入） | 专用输出信号（除上述公共输出） |
|---|---|---|
| `ads1298_acquire_ip` | `ecg_miso`、`ecg_ready_n`；`m_axis_ecg8_tready` | `ecg_mosi`、`ecg_sclk`、`ecg_cs_n`、`ecg_start`、`ecg_pwdn_n`、`ecg_reset_n`；`m_axis_ecg8_tdata[255:0]`、`m_axis_ecg8_tuser[127:0]`、`m_axis_ecg8_tvalid`、`m_axis_ecg8_tlast` |
| `ecg_quality_ip` | `s_axis_ecg8_tdata[255:0]`、`s_axis_ecg8_tuser[127:0]`、`s_axis_ecg8_tvalid`、`s_axis_ecg8_tlast`；`m_axis_ecg8_tready` | `s_axis_ecg8_tready`；`m_axis_ecg8_tdata[255:0]`、`m_axis_ecg8_tuser[127:0]`、`m_axis_ecg8_tvalid`、`m_axis_ecg8_tlast` |
| `ecg_classic_filter_ip` | `s_axis_ecg8_tdata[255:0]`、`s_axis_ecg8_tuser[127:0]`、`s_axis_ecg8_tvalid`、`s_axis_ecg8_tlast`；`m_axis_ecg8_tready` | `s_axis_ecg8_tready`；`m_axis_ecg8_tdata[255:0]`、`m_axis_ecg8_tuser[127:0]`、`m_axis_ecg8_tvalid`、`m_axis_ecg8_tlast` |
| `ecg_lead12_ip` | `s_axis_ecg8_tdata[255:0]`、`s_axis_ecg8_tuser[127:0]`、`s_axis_ecg8_tvalid`、`s_axis_ecg8_tlast`；`m_axis_ecg12_tready` | `s_axis_ecg8_tready`；`m_axis_ecg12_tdata[383:0]`、`m_axis_ecg12_tuser[127:0]`、`m_axis_ecg12_tvalid`、`m_axis_ecg12_tlast` |
| `ecg_qrs_ip` | `s_axis_ecg8_tdata[255:0]`、`s_axis_ecg8_tuser[127:0]`、`s_axis_ecg8_tvalid`、`s_axis_ecg8_tlast`；`m_axis_qrs_tready` | `s_axis_ecg8_tready`；`m_axis_qrs_tdata[63:0]`、`m_axis_qrs_tvalid`、`m_axis_qrs_tlast` |
| `ecg_packet_ip` | `s_axis_raw_tdata[255:0]`、`s_axis_raw_tuser[127:0]`、`s_axis_raw_tvalid`、`s_axis_raw_tlast`；`s_axis_lead12_tdata[383:0]`、`s_axis_lead12_tuser[127:0]`、`s_axis_lead12_tvalid`、`s_axis_lead12_tlast`；`s_axis_qrs_tdata[63:0]`、`s_axis_qrs_tvalid`、`s_axis_qrs_tlast`；`m_axis_pkt_tready` | `s_axis_raw_tready`、`s_axis_lead12_tready`、`s_axis_qrs_tready`；`m_axis_pkt_tdata[31:0]`、`m_axis_pkt_tkeep[3:0]`、`m_axis_pkt_tvalid`、`m_axis_pkt_tlast` |
| `ecg_wavelet_ip` | `s_axis_ecg8_tdata[255:0]`、`s_axis_ecg8_tuser[127:0]`、`s_axis_ecg8_tvalid`、`s_axis_ecg8_tlast`；`m_axis_ecg8_tready` | `s_axis_ecg8_tready`；`m_axis_ecg8_tdata[255:0]`、`m_axis_ecg8_tuser[127:0]`、`m_axis_ecg8_tvalid`、`m_axis_ecg8_tlast` |

上述 `ecg8` 与 `ecg12` 数据及 `TUSER` 位定义以 2.2 节为准。质量 IP 只改 `TUSER[87:56]`；导联 IP 只新增 `TUSER[99:88]`；经典滤波和小波 IP 不改 `TUSER`。在有效输出中，其他模块不得重写 `frame_seq`。QRS 的 64 位事件字按 3.E 节，打包的 32 位记录按 3.F 节。

### 2.5 七个 IP 的功能边界与交接条件

| IP | 必须实现的功能 | 必须交付给下一级的内容 |
|---|---|---|
| 采集 | ADS1298 复位与启动、ID/寄存器配置回读、SPI 连续读帧、状态字校验、8 路符号扩展、帧序号与丢帧计数 | 原始 8 路数值、24 位状态字、连续帧序号；异常时通过状态寄存器报告 |
| 质量 | 逐路导联脱落、饱和、平线、突变检测；阈值与窗口可配 | 数值原样通过，更新 4 类质量掩码 |
| 经典滤波 | 逐路低频基线高通、50 Hz 陷波、FIR 低通；三级独立旁路、定点饱和与暂态报告 | 8 路滤波数值，原样传递序号和质量字段 |
| 导联重建 | II/III 生成 I 与三个加压导联，重排 V1～V6，宽位计算与无效传播 | 12 路定序数值、12 位无效掩码、原帧序号 |
| QRS | QRS 专用增强、候选峰、自适应阈值、不应期、多导联融合、RR 间期 | 仅在检出时输出 64 位节拍事件；序号指向原始 R 峰帧 |
| 打包 | 三输入独立缓冲、按序输出类型化记录、DMA 块结束标记、拥塞计数与尾块刷新 | 单路 32 位 AXI4-Stream，`TKEEP=4'hF`，可由软件按长度解析 |
| 小波（可选） | 窗口化分解、阈值处理、重构、边界处理与延迟对齐 | 8 路小波处理数值，序号和质量字段保持不变 |

**导联无效传播约定**：任一质量掩码在 CH1 或 CH2 置位时，I、II、III、aVR、aVL、aVF 对应的导联无效位按源通道依赖关系置位；WCT 来源为 RA、LA、LL，故任一来源异常时 V1～V6 也置无效。胸导联本身的对应 CH 异常时，仅该胸导联置无效。采集 IP 的 WCT 寄存器回读失败时不得启动数据流，因此不另设未定义的“WCT 有效”旁路信号。

**现有采集 IP 与本文的差异**：当前 `ads1298_acquire_ip` 已实现共享增益配置，尚不支持八路增益分别设置；如系统需要逐路增益，A 组须增加八路独立配置寄存器及回读测试。此差异不影响已约定的 AXI4-Stream 数据接口。

## 3. 自定义 IP 清单与各组任务

### A 组：`ads1298_acquire_ip`（必做，采集负责人）

**职责**：配置 ADS1298，并将每次 `DRDY#` 对应的 SPI 输出变为一帧 `ecg8_axis`。模块内含 SPI 主机、独立 PWDN#/RESET# 上电复位状态机、寄存器写读与回读校验、连续采样状态机、216 位移位与字段解析、FIFO 和错误计数。

**外部端口**：100 MHz 时钟、`aresetn`、上述 7 根前端数字线、AXI4-Lite、`m_axis_ecg8`、可选中断。`ECG_MOSI` 接 ADC 的 `DIN`，`ECG_MISO` 接 `DOUT`，`ECG_READY` 接 `DRDY#`。SPI 极性/相位、片选保持、命令间隔与复位顺序均按 ADS1298 数据手册实现，不以普通 SPI 外设的默认时序代替。

**配置要求**：能读设备 ID；能设置 500/1000 SPS、8 路共用增益与正常输入/内部测试信号；能配置 WCT、RLD、导联脱落检测；能读回关键寄存器。如需逐通道独立增益，按 2.5 节扩展现有实现。第一版可以把经验证的寄存器表放 ROM，但必须能重发配置并验证回读。不得仅靠 ADC 复位默认值工作。

**帧要求**：检查 24 位状态字前导格式，按 CH1～CH8 顺序做 24→32 位符号扩展。`frame_seq` 每个 `DRDY#` 增加 1；SPI 未在下一次 `DRDY#` 前完成时计为丢帧/超时。**验收**：仿真连续 10 万帧无解析错误；测试正、负满幅和零值；故意阻塞下游后溢出标志及计数正确；板上优先使用 ADS1298 内部测试信号验证 8 路。

### B 组：`ecg_quality_ip`（必做，质量负责人）

**职责**：消费 `ecg8_axis`，输出同格式 `ecg8_axis`，只更新质量字段，不改采样值。结合状态字产生电极脱落掩码；按可配阈值检测接近满量程的饱和、连续近乎不变的平线、单点/短窗突变。每一路独立保存窗口状态。

**配置与验收**：阈值以 ADC count 和帧数配置，默认值需在 README 中解释；提供各类异常的独立计数。用正常正弦、断线状态字、全幅夹紧、常值输入和注入单点尖峰测试；要求误标与漏标的定义、预期输出掩码都写在测试向量中。仅靠 ECG 无法可靠还原运动伪迹，第一版只检测与标记。

### C 组：`ecg_classic_filter_ip`（必做，经典滤波负责人）

**职责**：输入/输出均为 `ecg8_axis`；8 路分别维护状态，内部包含三个可独立旁路的阶段：低频基线处理、50 Hz 陷波、FIR 低通。优先实现可稳定定点化的 IIR 高通＋二阶陷波＋线性相位 FIR。提供系数与使能配置，保留符号、序号和质量字段。

**约束**：形态保真路径的高通截止频率要低，首版以 0.05 Hz 设计目标并用定点仿真检查极点量化与启动暂态；另外可提供更强的“仅供显示”档位，但不得覆盖保真档。陷波应能旁路，必要时才启用。FIR 通带/截止频率可配置，不得把模拟端约 153 kHz 的截止频率误当作数字滤波器的实际性能。

**验收**：提供 Python/Matlab 浮点参考、定点参考与 HDL 对照；逐项测试 DC/低频漂移、50 Hz 正弦、宽带噪声、阶跃和模拟 ECG；说明各级增益、延迟、量化误差及资源消耗。全部阶段旁路时，数值必须与输入逐帧相同。

### D 组：`ecg_lead12_ip`（必做，导联负责人）

**职责**：输入处理后 `ecg8_axis`，输出 `ecg12_axis`。根据原理图，`CH1=II`、`CH2=III`，胸导联重排为 `V1=CH8`、`V2=CH3`、`V3=CH4`、`V4=CH5`、`V5=CH6`、`V6=CH7`。计算：

```text
I   = II - III
aVR = III/2 - II
aVL = II/2 - III
aVF = (II + III)/2
```

所有加减先扩展位宽，再进行除二和最终 32 位饱和；第一版除二采用二补码算术右移，负奇数按向负无穷舍入，并在测试说明中明确。`II/III` 的任一路无效时，I、aVR、aVL、aVF 都标无效；II、III 各自继承对应源通道的质量。胸导联按 2.5 节传播对应原始通道及 WCT 来源异常。**验收**：对已知 RA、LA、LL 人工电位及正/负边界值逐一比对 12 路输出；保证所有输出同一帧号。

### E 组：`ecg_qrs_ip`（必做，节拍负责人）

**职责**：从经典滤波后的 8 路输入独立生成节拍事件。IP 内部可采用 Pan–Tompkins 型流程：QRS 专用带通、差分、平方/能量、滑动积分、自适应阈值、不应期与多导联候选融合。QRS 专用带通只用于定位，不反向覆盖形态保真数据。[原始方法](https://pubmed.ncbi.nlm.nih.gov/3997178/)

**事件接口**：独立 AXI4-Stream，`TDATA[63:0]`：`[31:0]` 为 R 峰对应的原始 `frame_seq`，`[47:32]` 为 RR 间隔（采样点数），`[55:48]` 为参与确认的 8 路掩码，`[63:56]` 为事件标志（bit0＝RR 有效，其余预留）。事件仅在检测到 R 峰时发送，`TLAST=1`；不得把“事件发出时的帧号”误写成实际 R 峰的帧号。

**验收**：测试单导联干净、单导联断线、多导联噪声、连续双峰和无心搏窗口。提供 R 峰召回率、误报率、检测延迟与资源报告；第一版算法在与现有参考波形对齐后才能固定阈值。融合策略及默认选择的导联须写明。

### F 组：`ecg_packet_ip`（必做，输出与集成负责人）

**职责**：接收质量处理后的原始 `ecg8_axis`、`ecg12_axis` 和 QRS 事件流，合并为一条 32 位 AXI4-Stream 记录流，供现成 AXI DMA / FIFO 使用。`m_axis_pkt_tkeep[3:0]` 恒为 `4'hF`。三种记录可异步到达，靠原始帧号关联，不强行等待 QRS 事件。各输入设置 FIFO；拥塞时报告分流丢包计数。

**记录格式**：全部小端字节序的 32 位字。`word0=0x45434731`（ASCII `ECG1`）；`word1[31:24]` 为类型（1＝原始 8 路，2＝处理后 12 导联，3＝QRS 事件），`word1[15:0]` 为本记录总字数；`word2` 为对应原始帧号；`word3` 为 ADS 状态字，事件记录置零。`word4` 的四个字节依次为 8 路电极脱落、饱和、平线和突变掩码；`word5[11:0]` 为 12 导联无效掩码，其余置零。类型 1 的 `word6..13` 是 CH1～CH8；类型 2 的 `word6..17` 是 12 导联；类型 3 的 `word6` 是 RR 采样点数，`word7` 的低 8 位为参与导联掩码、次低 8 位为事件标志。记录总字数分别为 14、18、8。

事件记录的 `word3..5` 置零；原始 8 路记录的 `word5` 置零。`word1[23:16]` 保留且置零。`word2` 在事件记录中必须是 R 峰对应的原始帧号。

AXI4-Stream `TLAST` 默认每 **64 条完整记录**置位，不能在一条记录中间截断；可通过配置命令刷新不足 64 条的尾块。**验收**：用交错、缺失和背压输入测试解析完整性；每条记录长度、类型、帧号可由独立测试脚本复原；不能把现有 HDMI 的 AXI VDMA 当作 ECG 采集 DMA 使用。

### G 组：`ecg_wavelet_ip`（增强，不阻塞第一版）

**职责**：输入/输出均为 `ecg8_axis`，实现可定点验证的离散小波分解、阈值处理与重构，作为经典滤波的可切换对照路径。小波基、分解级数、阈值规则、窗口长度、边界延拓和固有延迟由负责人在详细设计中固定；先做 1 路，再扩展到 8 路复用。可以优先选择去基线或高频去噪其中一个目标，不要求同时解决所有噪声。

**验收**：与经典滤波对比信噪比、波形偏差、ST 段偏差、延迟以及 LUT/DSP/BRAM；输出同样保留序号和质量字段。由于小波窗口会增加延迟，切换路径时用帧号对齐，不能直接混合不同时间的帧。

## 4. Block Design 集成拓扑

```text
PmodA 7 根数字线
       │
       ▼
ads1298_acquire_ip ──► ecg_quality_ip ──┬──► ecg_packet_ip.s_axis_raw
                                         ├──► ecg_classic_filter_ip ──┬──► 路径选择器输入0
                                         │                             └──► ecg_qrs_ip ──► ecg_packet_ip.s_axis_qrs
                                         └──► ecg_wavelet_ip（可选）──► 路径选择器输入1

路径选择器输出 ──► ecg_lead12_ip ──► ecg_packet_ip.s_axis_lead12

ecg_packet_ip ──► AXI FIFO / AXI DMA ──► Zynq PS
AXI4-Lite 控制寄存器 ◄────────────────── Zynq PS
HDMI 视频链路沿用当前 ecg_system 工程
```

AXI Broadcaster、AXI Stream Switch、FIFO、DMA、AXI Interconnect、时钟和复位使用 Vivado 自带 IP；**只有表中 A～G 为本团队要写的自定义 IP**。Wavelet 路径接入时，用 Stream Switch 选择用于 12 导联输出的经典或小波路径；QRS 第一版固定取经典路径。此图只表示 PL 数据拓扑，不规定 PS 端的分析程序。

## 5. 分工、依赖与交付验收

| 负责人 | 独立交付物 | 依赖 |
|---|---|---|
| A 组：采集 | `ads1298_acquire_ip`、SPI 时序仿真、寄存器配置/回读表、测试信号记录 | TI 手册；PmodA 实际接线用于最终上板 |
| B 组：质量 | `ecg_quality_ip`、注入伪迹测试向量及阈值说明 | 统一 `ecg8_axis` 定义；可先用仿真源 |
| C 组：经典滤波 | `ecg_classic_filter_ip`、浮点/定点黄金模型、幅频/延迟报告 | 统一 `ecg8_axis` 定义；可先用仿真源 |
| D 组：导联 | `ecg_lead12_ip`、导联公式与边界值测试 | 统一接口；可先用仿真源 |
| E 组：QRS | `ecg_qrs_ip`、标注节拍测试向量和检出指标 | 统一接口；可先用仿真源 |
| F 组：集成 | `ecg_packet_ip`、DMA 数据解析脚本、BD 接线、约束、时序/资源总报告 | A～E 的接口版本；最终针脚映射 |
| G 组：增强 | `ecg_wavelet_ip`、与经典路径的同帧对比 | 不阻塞 A～F |

每组交付须包括：可综合 RTL/HLS 源码、可重复运行的仿真测试、黄金输出或参考脚本、IP 打包脚本/说明、寄存器表、已知限制、单 IP 资源和时序报告。统一目录建议为 `pl/ip/<ip_name>/{rtl,sim,ref,packaging}`。修改共用接口必须先更新本任务书并通知所有负责人，不能私自改变位宽或字段含义。

**集成顺序**：统一帧接口与仿真数据源 → A/B/C/D/E 并行开发 → F 组用仿真源验证打包 → A 组内部测试信号上板 → 接入真实前端 → G 组增强对比。第一版完成条件是 8 路连续采集、质量标记、经典滤波、12 导联、QRS 事件和打包输出全部通过独立仿真及整机连续运行；神经网络不属于本阶段验收。

## 6. 集成前必须补齐的资料

1. 前端板连接器到 PmodA 的**逐针脚表**，包括 7 根数字线、供电和地；再据此写 XDC。当前给出的原理图没有画出 PmodA 针脚映射。
2. ADS1298 初始化寄存器表：采样率、增益、WCT、RLD、导联脱落检测与内部测试信号的选项值，按实物和数据手册验证后冻结。
3. 参考测试数据和逐帧黄金结果。可选 500 Hz 的 [PTB-XL 十二导联数据](https://physionet.org/content/ptb-xl/1.0.2/)做算法验证；喂给 8 路采集链路前必须按本板 `II、III、V1～V6` 的通道顺序重排。
