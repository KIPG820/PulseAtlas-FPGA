# PulseAtlas A-board R1 implementation basis

- Board/part: PYNQ-Z2, `xc7z020clg400-1`.
- Programmable logic: unchanged R13 QRS-RAM + TFT hardware-SPI design.
- R13 implementation: routed with WNS `+0.151 ns`, WHS `+0.008 ns`, zero
  routing errors, and clean bitstream DRC.
- Display integration observed on PYNQ-Z2: the app started, the TFT IOP
  firmware loaded, and the first TFT frame was acknowledged.
- ADS1298 was disconnected during that R13 TFT-focused run; live ECG capture
  was not verified there.
- A-board R1 changes are software/package changes only: the local PS camera
  runtime and visual weights are removed and replaced by a reconnecting TCP
  client. They do not alter the bitstream, HWH, ECG data path, or TFT firmware.
- The A-to-B protocol and client have offline fake-server tests. No B service
  exists in this release, and the A-board package has not yet had a fresh
  physical board regression after repackaging.

Keep `ecg_frontend.bit` and `ecg_frontend.hwh` together. Their SHA-256 values
are recorded in `SHA256SUMS.txt`.
