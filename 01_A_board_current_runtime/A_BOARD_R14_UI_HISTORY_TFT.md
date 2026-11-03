# PulseAtlas A Board R14 — UI and TFT acceptance build

Date: 2026-10-27

This build contains the R14 application/UI updates and a matched routed
Vivado 2024.2 bitstream/HWH pair from `bld/ecg_nn_r14_live_status`.
The prior R14 archive accidentally included a stale hardware pair that omitted
the TFT SPI controller. This acceptance archive includes
`iop_arduino_tft_spi_0` at `0x44A10000`. The HWH comparison confirms the 51
previous modules are unchanged; the SPI controller is the sole module addition.

HDMI audio is deferred and is not a goal of this acceptance build. The ECG
acquisition and classifier path are unchanged. The already-resolved 5% capture
wait issue is not modified here.

## UI changes

- Larger AI result typography and waveform preview; removes repeated-run and
diagnostic-disclaimer copy while preserving the existing PS/PL comparison.
- ECG plots use extrema-preserving decimation so narrow QRS peaks survive the
  full visible window.
- The 12-lead page displays six seconds and identifies time direction, with
  concise labels that omit ADC-code and scaling explanations.
- TFT monitor, leads, focused-lead, and settings pages expose filter and gain
  controls; touch routes update the shared application settings.
- The PSD page has a dedicated dBFS/Hz color scale.
- History reads actual SQLite acquisition sessions and shows sample counts,
  quality flags, inference results, and incomplete/error state without sample
  rows.

## Validation

Offline UI, waveform, history, touch-route, and Python syntax regressions passed
in the PYNQ environment. Render checks covered 1280x720 HDMI and 320x240 TFT.

The Vivado route report shows 60,505 of 60,505 routable nets complete, no route
errors, WNS +0.210 ns, WHS +0.019 ns, and all specified timing constraints met.
Board-level TFT refresh/touch acceptance is pending the live run.

## Install

```sh
cd /home/xilinx/pulseatlas_a_r14_ui_20261027
sha256sum -c SHA256SUMS.txt
sudo -E /usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py
```
