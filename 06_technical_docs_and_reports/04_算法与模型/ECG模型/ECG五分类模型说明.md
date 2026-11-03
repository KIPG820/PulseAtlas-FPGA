# ECG 五分类模型说明

## 数据与任务

- 数据集：PTB-XL 1.0.3。
- 官方患者级划分：fold 1–8 训练、fold 9 验证/阈值选择、fold 10 最终一次测试。
- 输入：`[batch, 12, 5000]`，500 Hz、10 秒、标准 12 导联顺序。
- 输出：NORM、MI、STTC、CD、HYP 五个独立 logits，属于多标签分类。

## 网络结构

- Stem：12→16，kernel 15，stride 4。
- Stage 1：两组 16 通道残差块。
- Stage 2：16→32，首块 stride 2，再一组 32 通道残差块。
- Stage 3：32→64，首块 stride 2，再一组 64 通道残差块。
- 自适应平均池化、dropout 0.2、64→5 全连接。
- 每个残差块使用 kernel 7 的两层 Conv1d，必要时用 1×1 投影匹配维度。

## 训练增强

幅度缩放 0.9–1.1、基线漂移、50 Hz 工频、Gaussian 噪声和 20 ms 时间平移。训练配置与源代码已收录。

## 结果

- fold 10：macro-AUROC 0.9312，macro-AUPRC 0.8273。
- FP32→INT8 PTQ 在 fold 9 上 macro-AUROC 仅下降约 0.00049，macro-AUPRC 下降约 0.00536。
- 硬件中使用的 INT8 权重/偏置/scale/zero-point 合同与 HLS 参数均保留在源码材料中。

该结果是公共数据集科研/竞赛评估，不代表临床诊断性能。
