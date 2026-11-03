# PulseAtlas 最新完整联测包（给队友，2026-10-22）

本包对应已在本机测试的 Fusion_Contrast_20261022，第14轮微调模型；第一阶段原姿态、标定与位置规则保持原样。默认开启连续电极身份稳定。请完整解压后运行，不要只拷贝PT权重。

## 1. 首次配置（Windows + Anaconda）

在 Anaconda Prompt 中进入【解压后本文件所在目录】。可在资源管理器复制文件夹路径，用 `cd /d "你的解压目录"` 进入。路径可自由选择，不需要本机的D盘工程。

```bat
conda env create -f environment.yml
conda activate ecg-yolo11
python verify_package.py
python app/camera_trial.py --check
```

创建环境需要联网下载依赖；`--check`是离线启动/模型/录像自检，不打开摄像头。已有兼容环境可直接激活使用：Python3.11、PyTorch2.5.0+cu118、Ultralytics8.3.163、OpenCV4.12；完整版本在requirements-conda.lock.txt。NVIDIA显卡需可用驱动；程序自动选择CUDA，不可用时回退CPU，CPU会更慢。

## 2. 打开USB摄像头

```bat
conda activate ecg-yolo11
02_start_camera.bat
```

在界面选择摄像头编号，分辨率保持1920×1080，再点击“开始摄像头”。默认编号0；若打开内置摄像头，改试1或2。先关闭Windows相机/会议软件，避免占用设备。每次只开一个测试窗口。

- **02_start_camera.bat**：默认推荐，一阶段 + 新微调电极模型 + 连续身份稳定。
- **03_previous_baseline.bat**：原上一版主模型，无新增身份稳定，用于对照。
- **04_connector_only.bat**：新模型 + 身份稳定，仅看电极检测，不运行姿态阶段。
- **05_raw_detector.bat**：新微调模型原始单帧结果，关闭身份稳定，用于定位问题。

旧界面名称：R=RA、L=LA、N=RL、F=LL、C1～C6=V1～V6。尤其C4=V4、N=RL。默认先看清晰电极，连续3帧确认后输出稳定身份；灰色Pending ID表示身份待确认，不参与匹配成功判定。停止并重新开始会清空轨迹。

## 3. 保存测试结果

误判时点“保存当前误例”，会保存无框raw.jpg、显示结果和JSON；“开始/停止记录”保存原始录像及逐帧预测。所有新结果均在本包runtime目录生成，无需访问训练工程。JSON同时保留原始预测、最终身份、时间戳及待确认状态；反馈时把相应快照/会话文件夹发回即可，不用额外填表。

## 4. 包内资源

- models/connector_selected.pt：默认GUI使用的第14轮模型（旧枚举）。
- models/electrode_comparison.pt：同参数规范十类名模型，供自行接入；裸PT不包含连续身份稳定。
- models/connector_previous.pt：原模型对照。
- app/vendor_pose/：第一阶段原ONNX、实现、标定资源。
- app/temporal_identity.py：当前帧候选上的连续身份稳定；不输出不存在的历史位置。
- reports/：训练选择、检测回归、一二阶段离线联测与255帧时序复现证据。
- docs/TRAINING_AND_RESULTS_CN.md：完整训练方法与已知限制（其中原工程路径仅为来源记录）。
- training_code/：实现参考；重训仍需要原工程数据，运行本测试包不需要训练集。

## 5. 当前验证范围

旧全类Test14为59TP/0FP/0FN；新对比静态Test4中V4、RL各3/4正确且无冲突，反光帧仍有错误。默认连续回放在四个已标注时刻附近V4为4/4、RL为3/4（1处待确认）。这些是同一段录像的回顾结果，不能保证其他人物、光照和遮挡下零误判。身份稳定依赖清晰初始识别及可靠轨迹，交叉/遮挡时可能待确认。

输入请求1080p/30FPS；本机已有离线完整链路约17.4FPS（无USB/Tk显示），30FPS文件头不等于实际处理30FPS，队友现场性能需实测。界面位置结果仅相对原标定参考点，不验证电气连通或解剖贴放。

本包不包含训练图像、真人录像、历史运行数据或虚拟环境本体。首次解压无需训练即可运行；请保留SHA256SUMS.txt以校验资源完整性。
