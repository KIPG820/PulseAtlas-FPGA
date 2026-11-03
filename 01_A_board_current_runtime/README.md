# PulseAtlas A Board R14 — UI, history, and TFT controls

R14 updates the application UI and history presentation. This acceptance build
pairs the R14 application with the routed Vivado overlay and its matching HWH.
The hardware description adds the missing Arduino IOP TFT SPI controller at
0x44A10000; every other overlay module is unchanged from the prior application
pair. ECG acquisition and the neural-network path remain unchanged.

R6 keeps the R5 ECG, display, touch, and button behavior unchanged and adds
the B-board electrode-presence results to the Vision page. HDMI shows all ten
calibrated sites (RA, LA, RL, LL, V1–V6), each marked recognized, review, or
pending. TFT shows the recognized count and names the sites that need review
or are still pending. A stale result is marked expired rather than presented
as current. This is visual pad-presence guidance near the calibration points;
it does not measure skin contact or electrical continuity.

This release packages the current PYNQ-Z2 A-board system: ECG capture and
classification, the LVGL interface, HDMI, ST7789/GT911 TFT and touch, four
pushbuttons, and SW0 display selection. TFT and HDMI now share working display
filter and waveform-gain settings, with a settings shortcut available from the
TFT header and the HDMI monitor. This archive pairs the R14 ECG/TFT feature set
with an R13 HDMI-audio bitstream. ECG, network, and USB camera indicators update
while the application is idle, without opening their monitor or inference page.

R13 adds HDMI LPCM audio packets and a short tone on each accepted hardware QRS
event. Audio is only expected from an HDMI audio-capable monitor/receiver with
working speakers or an audio output; the TFT path remains silent. The QRS tone
is an event cue and does not change model inference or indicate a diagnosis.



## ECG capture DMA re-arm fix

AI batch capture previously decoded every packet in a completed DMA block
before submitting the next S2MM transfer. Python decoding runs while the PL
packetizer is producing data, so this created an avoidable period with no DMA
buffer armed and could fill the finite packet FIFOs. The capture path now copies
the completed words and re-arms DMA immediately, then parses the copied block.
On every exit path it stops the final in-flight transfer before freeing its
buffer. A regression test checks this ordering.

R9 also closes the startup-only loss window. The first S2MM buffer is armed and
the continuous DMA drain worker is running before ADS1298 streaming is enabled.
Previously the ADC could fill the first packet block before the worker existed,
which caused FIFO backpressure and a burst of dropped frames at monitor start.
The change does not alter the FPGA image, sampling rate, filters, gain, lead
mapping, model input, or inference thresholds.

## ECG inference window fix

R7 preserves the filter warm-up (3,000 frames) and the model input (exactly 10,000 12-lead frames). It retains a 10-second lookback and selects the newest contiguous, quality-clean window from recent acquisition history. A transient flag in only the newest window no longer blocks inference if an earlier valid window is still recent. The selector never fills missing samples or ignores quality flags; real sequence gaps and flagged frames remain ineligible. The timeout separates the latest-window cause from the ADC drop-count increase during this attempt, instead of showing a lifetime counter as if it were new loss. The selected window age and counter deltas are also written to the terminal log.

The packaged `ecg_frontend.bit` and `ecg_frontend.hwh` come from the same
Vivado 2024.2 routed project. Keep this pair together. The earlier R14 archive
accidentally carried a stale hardware pair without the TFT SPI controller;
this acceptance archive corrects that pairing. HDMI audio remains deferred.
The application does not load PYNQ Base or a second overlay. ECG acquisition stays stopped until
**实时监测** is opened and started.

While monitoring is stopped, the ADS1298 remains powered in SDATAC and its ID
is polled every second. Starting ECG monitoring starts only the data stream;
stopping it returns the converter to idle polling without resetting it. If the
chip is disconnected, the status clears and the hardware retries automatically.
The `NET` indicator covers a live wired or wireless link and is polled every
500 ms. The USB indicator checks local `/dev/video*` nodes and receives passive
camera status from B; install the matching B R2 service for the remote camera
indicator. Passive status does not start image capture or inference.

## R14 UI, waveform, and history corrections

The AI page removes repeated-run/equality and diagnostic disclaimer text,
increases result typography, and gives the 12-lead preview more space. The
selected backend and its existing three-pass mean-comparison behavior remain
unchanged. This release does not change inference-window selection, quality
checks, preprocessing, model input, or classification logic.

ECG traces use extrema-preserving decimation so narrow QRS complexes are not
lost when a long window is reduced to screen pixels. The 12-lead page displays
the complete recent six-second window and states the time direction. TFT
monitor, lead, and settings pages expose the same filter bypass and
waveform-gain controls.

The frequency page reserves a separate color scale for PSD in dBFS/Hz, keeping
its units clear of the frequency and time axes. The history page reads actual
SQLite acquisition sessions and displays stored sample counts, quality status,
and inference results. Database errors and incomplete recordings are shown as
such; the UI does not create sample history rows.

## Board controls

BTN0/BTN1 move the selection or switch pages, BTN2 selects/starts/stops, and
BTN3 returns to the parent page. On the AI page, BTN0/BTN1 select **PS NumPy**
or **FPGA + DMA**; BTN2 runs three inferences on the selected backend. The
same controls are available through GT911 touch.

SW0=0 blanks HDMI with a black frame while keeping its video scanout alive, so
the UI is visible only on TFT. SW0=1 restores the current UI on both HDMI and
TFT without closing and restarting the HDMI pipeline. The switch is read from
the matching R13 audio-enabled overlay. If the TFT IOP cannot start, HDMI can still run and
the failure is reported in the terminal.

## Install and run

Copy this archive to the PYNQ-Z2 and extract it to a new directory, for example
`/home/xilinx/pulseatlas_a_r14_ui_20261027`. From a board terminal:

```sh
cd /home/xilinx/pulseatlas_a_r14_ui_20261027
sha256sum -c SHA256SUMS.txt
sudo -E /usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py
```

To save the log:

```sh
sudo -E /usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py 2>&1 | tee pulseatlas_a.log
```

The app loads the bundled R14 overlay and TFT IOP firmware, then runs the ECG
golden-vector check in the background. Connect the ECG frontend before starting
monitoring. `Ctrl-C` exits and releases the display and DMA buffers.

## Vision page and B board

The TFT header uses the original ECG logo scaled uniformly to fit. The TFT
monitor page includes a filter toggle and gain minus/plus controls; its settings
page exposes the same controls. Turning display filtering off shows raw
packetized ECG samples, converted into the FPGA's 12-lead order. AI inference
continues to use the standard filtered lead stream in either display mode.
Gain changes affect the rendered waveforms on both screens and do not alter
acquisition or model inputs. The Vision page on A connects to the service endpoint in
`vision_peer.json` (default `192.168.2.100:8765`). This address is a placeholder;
set it to B's address before connecting to the B R2 service. It can also be
overridden with
`PULSEATLAS_B_HOST` and `PULSEATLAS_B_PORT`. The A client reconnects in the
background and retains only the latest decoded preview frame. If B is absent,
the page reports that state without affecting ECG, HDMI, TFT, or touch.

The B service reports USB device presence from its configured `/dev/videoN`
node even while capture and inference are stopped. A keeps this status
subscription active in the background; opening the Vision page starts capture
and electrode-presence inference.
The wire format and exact status/result fields are defined in
[`VISION_B_SERVICE_PROTOCOL.md`](VISION_B_SERVICE_PROTOCOL.md). The saved
manual torso/chest calibration JSON files are retained under
`vision_guide/calibration/` for the B integration. This package intentionally
contains no camera driver code, OpenCV model runtime, or visual-model weights.
The HDMI and TFT pages render B's preview and status; B remains responsible for
freshness, confidence, pose validity, and marker correctness.

## Validation

Run the offline regressions on the board or a compatible Python environment:

```sh
for test in build_tools/test_capture_dma_rearm.py build_tools/test_ecg_display_metrics.py build_tools/test_history_database.py build_tools/test_lead_off_contract.py build_tools/test_nn_timing.py build_tools/test_sw0_display_mode.py build_tools/test_tft_mailbox.py build_tools/test_tft_ui.py build_tools/test_touch_routes.py build_tools/test_vision_service_client.py; do
  sudo -E /usr/local/share/pynq-venv/bin/python "$test" || exit 1
done
```

The vision client test starts a local fake B service. It checks status mapping,
JPEG decode, start/stop transport, and the offline/reconnect state. It does not
require B hardware or open a camera.

The bundled hardware pair comes from `bld/ecg_nn_r14_live_status`. Its routed
report shows 60,505/60,505 routable nets completed, no routing errors, WNS
+0.210 ns, WHS +0.019 ns, and all specified timing constraints met. Comparing
its HWH with the previous package shows exactly one added module:
`iop_arduino_tft_spi_0` (`axi_quad_spi`, base `0x44A10000`); all 51 prior
modules and versions match. This validates the hardware pairing offline. Board
TFT refresh and touch still need visual acceptance after launch. HDMI audio is
deferred in this build.

## Package contents

- `ecg_frontend.bit` + matching `ecg_frontend.hwh`: the routed R14
  ECG/HDMI/TFT pair with the TFT SPI controller. HDMI audio is deferred.
- `pulseatlas_r9_app.py`, ECG drivers, LVGL and TFT renderers, and the
  8-KiB TFT IOP image: application/runtime components.
- `model_ptq_int8_hw.json`, `model_ptq_int8_weights.npz`, normalization, and
  golden vectors: the existing ECG classifier contract and regression data.
- `vision_service_client.py`, `vision_peer.json`, and
  `VISION_B_SERVICE_PROTOCOL.md`: A-side B-service adapter and interface.
- `vision_guide/calibration/*.json`: restored manual calibration artifacts for
  later B integration.

See `HDMI_AUDIO_R13.txt`, `R14_LIVE_STATUS.txt`, `R13_TFT_SPI.txt`, and
`build_tools/IMPLEMENTATION_REPORT.md` for implementation and validation notes.
