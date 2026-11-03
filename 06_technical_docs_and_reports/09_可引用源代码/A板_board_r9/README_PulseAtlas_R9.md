# PulseAtlas + R9 HDMI/TFT SW0 display mode

This package starts PulseAtlas as a standalone application on the PYNQ-Z2. It draws the UI directly through the R9 bitstream's HDMI output, reads ECG packets from the existing frontend DMA, and sends approved 10-second windows to the existing PS and PL neural-network paths. The Vision page runs one YOLO11n-pose model on the PS CPU through OpenCV DNN and projects your saved manual electrode calibration; the ECG bitstream, PL inference path, and ECG DMA are not changed. Notebook controls are not used.

The board and Windows preview use the same LVGL 9.4 page builders, logo asset, and generated fonts from `ui/pulseatlas/lvgl_pc/main.c` for the ECG pages. The board package includes `libpulseatlas_lvgl.so`, cross-compiled for the PYNQ-Z2 ARM hard-float Linux ABI, and `pulseatlas_lvgl_backend.py`, which passes real ECG samples, heart rate, device status, AI results, and physical-key focus into those pages. LVGL renders RGB565 into its own persistent frame; Python converts a completed frame to the BGR24 format expected by PYNQ `VideoOut` and copies it into the HDMI DMA buffer. The Vision page is rendered in Python so the camera image, pose outline, and calibrated lead markers can be composed without changing the LVGL ECG pages or the ECG PL data path.

The original R9 bitstream did not contain a GPIO for the four pushbuttons, so software polling alone could not read them. This button-enabled build adds one four-bit AXI GPIO input in the PL control map at 0x40007000; it uses the PYNQ-Z2 button pins D19, D20, L20 and L19 already listed in ecg_system.xdc. ECG/NN data-path logic is unchanged.

## Four-button controls

The keys are debounced in software (45 ms stable state) and trigger once on a press edge:

| Button | Home | Page / monitor | 12-lead view |
| --- | --- | --- | --- |
| BTN0 | Previous card | Previous page | Previous lead |
| BTN1 | Next card | Next page | Next lead |
| BTN2 | Open selected card | Start/stop on Monitor; run inference on AI | Open selected lead |
| BTN3 | — | Return to Home | Return to 12-lead overview |

On the **Vision** page, entering the page opens the USB camera and starts PS CPU inference. BTN2 starts or stops it. Leaving Vision keeps active camera inference running in the background; return to Vision and press BTN2 to stop it. App shutdown releases the camera. BTN3 returns Home. The visual task uses a dedicated camera-capture thread and a separate YOLO inference worker. Camera frames continue to arrive while the model runs; the annotated HDMI preview refreshes at up to 20 Hz, while the UI reports measured preview refresh, camera capture, model inference FPS, and single-inference latency separately. Inference remains the original 256x192 FP32 ONNX model, has no artificial one-per-second limit and is scheduled at Linux `nice 10` using up to two OpenCV/TBB threads. On this dual-core Cortex-A9, a YOLO pose pass takes about 2.3 seconds in the current board test, so the model itself runs around 0.4 FPS; 30 neural inferences/s cannot be reached by PS/Python scheduling changes. ECG and UI threads retain normal priority. Check ECG packet integrity and UI responsiveness during concurrent use.

The highlighted card or lead shows the current selection. On the AI page, BTN2 starts the comparison; the visible **开始推理** control is also clickable when using a pointer/touch input. Returning Home does not stop active capture; stop it from the Monitor page before shutting down.

## Landscape TFT mirror

The matching `ecg_frontend.bit` and `ecg_frontend.hwh` also include the PYNQ Arduino MicroBlaze IOP used by the ST7789/GT911 panel. When `tools/tft_pulseatlas_iop.bin` is present, the app starts the TFT mirror on the same R9 overlay; it does not load PYNQ Base or replace the ECG/HDMI design. The TFT uses a compact 320×240 landscape layout for the same app pages and live state. Touch navigation calls the same page/action handlers as the HDMI UI. The **视觉** page mirrors the annotated camera frame and the current camera/inference status, including FPS, latency, pose status and confidence. Its refresh rate is limited by the TFT serial transfer, so it will be lower than the HDMI preview rate.

The bottom row gives direct access to Home, Monitor, 12-lead, Spectrum, AI and Vision. History and Settings remain available from Home. GT911 touch uses the verified 270-degree landscape mapping by default; set `PULSEATLAS_TFT_ROTATION=90` only if the panel responds in the opposite direction. If the IOP binary is absent or initialization fails, HDMI continues to run and the terminal reports the TFT error.

### SW0 display mode

The PYNQ-Z2 slide switch SW0 selects the active displays while PulseAtlas is running. Set **SW0=0** for TFT only; the app closes HDMI output and skips its 1280×720 rendering, leaving ECG acquisition, AI, vision, TFT drawing, and touch active. Set **SW0=1** for HDMI and TFT together. Switching back to 1 reopens HDMI and redraws the current page. The GPIO is debounced in software. This release's bitstream and HWH must stay paired; an older bitstream without `sws_gpio` falls back to HDMI on. For bench diagnosis only, `PULSEATLAS_HDMI_MODE=on` or `off` overrides the switch.

The TFT backend sends only changed 16×16 regions after the first full frame. Its ST7789/GT911 IOP uses the measured short reset sequence; a 320×240 test frame took about 0.95 s and a 40×40 update about 0.17 s on this board. Dynamic page rate depends on how much of the image changes, and taps made while the shared display/touch data lines are writing pixels can be delayed.

## Start on the board

Copy and extract the complete package to the PYNQ-Z2, for example under `/home/xilinx/pulseatlas_r9`. Connect the board's HDMI output to a 1280 x 720 display. Then run this from a board terminal:

```sh
cd /home/xilinx/pulseatlas_r9
sudo -E /usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py
```

The matching R9 overlay is loaded by the application. PulseAtlas appears on TFT, and on HDMI when SW0=1; fixed NN golden-vector checks run in the background. ECG acquisition stays stopped until the user opens **实时监测** and starts it. The USB keyboard fallback supports 1 monitor, 2 12-lead, 3 AI, 4 spectrum, H home, Space start/stop acquisition, A run AI, and Q quit. Ctrl-C also shuts down the application.

The LVGL font glyphs and PulseAtlas mark are embedded in the ARM shared library; the board does not need a separate font installation for the main UI. The Vision page uses the bundled CJK font and a PS-rendered camera view. It bundles `vision_guide/models/yolo11n-pose-cv454.onnx`, a full-precision 256x192 compatibility graph from the YOLO11n-pose checkpoint. The camera is captured at 640x480 and letterboxed down to the model input; weights remain unchanged. The graph exposes raw box, person-score, and COCO17 pose heads before OpenCV 4.5.4's incompatible dynamic-shape decode tail; DFL box decoding, person NMS, and keypoint transforms run in NumPy. Only this one model is loaded for board vision. The package also includes the original manually calibrated version 2 torso and chest JSON files. Inference uses OpenCV's CPU backend and does not need PyTorch or Ultralytics on the board.

Before entering Vision, verify the board's PYNQ Python environment can load the camera and ONNX runtime:

```sh
sudo -E /usr/local/share/pynq-venv/bin/python -c 'import cv2, numpy; print("OpenCV", cv2.__version__, "DNN", hasattr(cv2, "dnn"))'
ls -lh vision_guide/models/yolo11n-pose-cv454.onnx vision_guide/calibration/*.json
sudo -E /usr/local/share/pynq-venv/bin/python build_tools/test_ps_vision_model.py
```

Run the packaged board-free navigation, timing, electrode-contact, and PS-worker checks before opening the HDMI app:

```sh
for test in build_tools/test_button_navigation.py build_tools/test_touch_routes.py build_tools/test_lead_off_contract.py build_tools/test_nn_timing.py build_tools/test_ps_vision_runtime.py; do
  sudo -E /usr/local/share/pynq-venv/bin/python "$test" || exit 1
done
```

Then verify the actual USB camera and shoulder/hip pose before launching the full app. Stand in view with your upper body and lower abdomen visible:

```sh
sudo -E /usr/local/share/pynq-venv/bin/python build_tools/test_ps_vision_model.py --camera --frames 10 --require-torso
```

This prints per-frame torso detections, shoulder/hip coordinates, PS inference latency, and measured camera-loop FPS, then releases the camera. It also checks that both manual calibration files are present. The test does not load or access the ECG overlay or DMA. If the camera test passes, start the full UI as below.

The PS implementation preserves the model weights and captures 640x480 frames before reducing the inference input to 256x192. On this PYNQ-Z2, a blank-frame model pass measured 3.19 seconds at one OpenCV thread and 1.83 seconds at two threads, with 134.7 MiB peak RSS in the two-thread benchmark. A saved reference image's shoulder/hip markers differed by at most 8.42 pixels from the 640x480 model. The camera was not connected during this measurement, so live-camera detection, HDMI behavior, and ECG packet integrity under concurrent inference remain unverified. The live page reports measured result-update FPS and per-frame inference time.

To launch PulseAtlas automatically at board startup after the manual HDMI test passes, copy `pulseatlas-r9.service.example` to `/etc/systemd/system/pulseatlas-r9.service`, then run `sudo systemctl enable --now pulseatlas-r9.service`. This sample uses the package path above and runs as root so it can access the PL overlay, HDMI, and USB input events. Use `sudo systemctl disable --now pulseatlas-r9.service` to turn autostart off.

## First live test

1. Wait for the AI golden-vector status to show **通过**. A failure keeps live AI inference disabled and is shown on the AI page.
2. Connect electrodes, open **实时监测**, then press **开始监测**. The live ECG, QRS-derived heart rate, 12-lead view, spectrum, and heart-rate trend are all sourced from R9 capture data.
3. Open **AI 分析**, choose PS NumPy or FPGA + DMA, and press **开始推理** (or press BTN2). The app preserves the 3,000-frame filter warm-up and exact 10,000-frame model input. It searches the recent 20 seconds of stable acquisition for the newest window with continuous frame IDs and zero hardware quality flags, allowing up to 10 seconds of lookback when a transient bad frame enters the newest window. Real frame gaps and quality flags are still rejected; samples are never interpolated or repaired. The selected backend then runs three times. Select the other backend to reuse the same cached input for a fair latency comparison. The page reports average latency and output consistency. Historical ADC drop counts and packetizer flow-control stalls are diagnostic counters, not grounds to reject an otherwise clean window. If no clean window arrives, the error distinguishes the latest window issue from ADC frames newly dropped during that attempt.
4. Press **停止监测** to stop acquisition while leaving PulseAtlas on HDMI. Use `Q` or `Ctrl-C` to close the app and release the HDMI/DMA buffers.

If the AI page cannot find a valid window, the terminal prints a `PulseAtlas AI window timeout:` line with ADC and pipeline counter deltas, recent gap steps and quality-flag counts, RX DMA block delta, and any worker error. The on-screen timeout reports the latest-window cause separately from the ADC drop-count delta during this attempt; an old cumulative drop count is never presented as new loss. An increasing ADC `frame_errors` count points toward SPI/header trouble; increasing ADC `drop_count` plus packet backpressure suggests a downstream stall; a clean ADC delta with lead-frame gaps narrows the fault to later pipeline/DMA stages. These are diagnostic clues, not proof of a cause without board measurements. The AI page uses a lightweight display preview; inference still uses all 10,000 lead frames. To save the line, launch with `sudo -E /usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py 2>&1 | tee pulseatlas_capture.log`. The R10 build prints `PulseAtlas R9 pose/manual calibration PS vision build 2026-10-06 r10` at launch so the active package can be identified.

The R10 Vision page keeps the USB preview moving independently of YOLO inference. The displayed overlay is based on the last completed inference frame; its age is shown because the model may take more than two seconds per pass. The Vision page projects your saved manual electrode markers using the detected shoulder-to-hip torso pose; it does not verify electrode contact or clinical position. Its included JSON files are the restored manually calibrated version 2 values. The History page's database is still not persistent. The Settings page reports the current R9 capture status. The 12-lead classifier is an engineering demonstration, not a clinical diagnosis.

## Rebuilding the button-enabled overlay

build_tools/add_pynq_buttons.tcl applies the block-design change to a copy of the R9 Vivado project. The source project already has the four button pin constraints in ecg_system.xdc. Regenerate the wrapper, rebuild the bitstream, and keep the new .bit paired with the matching generated .hwh; do not use the old R9 bitstream with this button-enabled app.

The delivered button-enabled overlay was implemented with Vivado 2024.2 for `xc7z020clg400-1`. Post-route DRC reported 0 errors; all routable nets completed routing. The timing summary reports WNS `+0.018 ns`, TNS `0`, WHS `+0.009 ns`, and THS `0`, so the user-specified setup and hold constraints are met, with very little setup margin. Vivado still reports that the HDMI `SerialClk` generated-clock constraint has no logical path from its PXL master clock; the timing summary does not count this as a user-constraint violation. Verify HDMI output and button operation on the physical PYNQ-Z2 before treating the board build as validated.

## Rebuilding the board LVGL library

From the Windows project checkout, run `python ui/pulseatlas/lvgl_pc/build_board.py` with the Vitis 2024.2 ARM GNU toolchain installed at its default path. The script writes `ui/pulseatlas/board_r9/libpulseatlas_lvgl.so`; set `PULSEATLAS_ARM_GCC` to a different `arm-linux-gnueabihf-gcc` path if needed. `check_board_renderer.py` builds a host-side test library and renders every LVGL page to PNG for visual regression checks without PYNQ hardware.

## Files and bitstream pairing

Keep the button-enabled ecg_frontend.bit and matching ecg_frontend.hwh together with the R9 drivers, model contract, weights, normalization, and golden_vectors/. Do not mix them with R7/R8 or the later R9Q package.
