# ECG packet FIFO BRAM prefetch trial — 2026-10-11

Status: functionally passed bounded RTL simulation; rejected for integrated candidate due to poor resource tradeoff.

The current packet IP contains 384x64, 512x64 and 64x64 FIFOs. The candidate preserves the reference acceptance rules (a push while full is rejected even when pop is also asserted), count, flags and the first-word-visible output whenever nonempty. Synchronous next-head prefetch plus same-address write forwarding makes the RAM eligible for block mapping. No formal proof or packet-system-level regression has been performed.

Vivado xsim tested 512x64, 384x64, 64x64 and 32x4, each for 20,000 cycles. Tests include forced fill/full, drain/empty, simultaneous push/pop, randomized data and periodic reset. All valid outputs and status matched cycle by cycle. See xsim.log and tb_fifo.sv.

Vivado 2024.2 out-of-context synthesis, xc7z020clg400-1, 512x64:

| Design | LUT | FF | BRAM tiles |
|---|---:|---:|---:|
| Existing asynchronous FIFO | 707 | 19 | 0 |
| Synchronous prefetch + forwarding | 547 | 532 | 7.5 |

Saving 160 LUT costs 513 FF and 7.5 BRAM tiles. Current integrated placement already uses 122/140 BRAM tiles and 100% slices. This tradeoff is too weak to prioritize and is not merged into the integrated design. Do not apply a global ram_style change to the ECG FIFOs. Narrower serialized packing would require a different interface/microarchitecture and additional verification.

Next architecture priorities: reduce shared vision accelerator control/requantization logic and examine replacement of the 3,440-LUT/16-BRAM Arduino IOP subsystem with a compact controller while preserving TFT/touch timing. Existing release and board files were not modified; no board upload occurred.

Sources and reproducible synthesis script are in pl/packet_fifo_prefetch/. Source reference is bld/ecg_nn_r9_tft_lean/ip_repo_ecg/ecg_packet_ip/src/ecg_sync_fifo.sv.
