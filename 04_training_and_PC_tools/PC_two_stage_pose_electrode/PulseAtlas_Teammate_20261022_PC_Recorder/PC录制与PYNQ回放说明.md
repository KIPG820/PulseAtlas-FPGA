# PulseAtlas PC 推理录制与 PYNQ HDMI 回放

## 1. 在 PC 上录制

1. 双击 `06_record_labeled_demo.bat`。
2. 选择外接 USB 摄像头编号和分辨率，点击“开始摄像头”。
3. 等待人体姿态和电极检测画面稳定。
4. 点击“录制演示视频”，完成展示动作后点击“停止演示录制”。
5. 点击“打开结果目录”，进入 `runtime/demo_exports/demo_日期_时间_xxxxxx/`。

目录包含：

- `pulseatlas_pc_offline_demo.mp4`：1280×720、30 FPS 的完整界面视频；
- `source_results.jsonl`：每个实际推理帧的检测结果和时间戳；
- `demo_metadata.json`：模型、录像时长、编码帧数和用途说明。

“记录原始数据”仍保留原有功能，只记录摄像头原图和预测 JSON。它与“录制演示视频”互不影响。

## 2. 复制到 PYNQ A 板

把视频和 `pynq_playback/play_offline_demo.py` 上传到 A 板，例如：

```bash
/home/xilinx/pulseatlas_r9/demo/pulseatlas_pc_offline_demo.mp4
/home/xilinx/pulseatlas_r9/demo/play_offline_demo.py
```

停止当前占用 HDMI 的 PulseAtlas 程序，再播放：

```bash
sudo pkill -f run_pulseatlas_r9.py || true
cd /home/xilinx/pulseatlas_r9
sudo -E /usr/local/share/pynq-venv/bin/python \
  demo/play_offline_demo.py \
  demo/pulseatlas_pc_offline_demo.mp4 \
  --board-dir /home/xilinx/pulseatlas_r9 --loop
```

按 `Ctrl+C` 停止播放。随后按原有方式重新启动 PulseAtlas 主程序。

## 3. 画面含义

视频中的检测框、置信度和位置重合状态来自 PC 端双模型推理。PYNQ 播放脚本只负责解码视频并通过 HDMI 输出，不重新执行视觉模型。
