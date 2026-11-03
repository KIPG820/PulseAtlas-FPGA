# B board R19: measured transfer and scheduling experiments

The deployed B-board service uses the R16 depthwise bitstream and the R19
`variant_both` INT8 backend. It retains the R15 pose network, calibration,
electrode presence model, HSV color matcher, TCP protocol, and 5 FPS preview cap.
The service remains enabled at boot. The previous service unit is saved on B at
`/home/xilinx/PulseAtlas_B_Vision_R19_UncachedSync/service_before_r19.service`.

## Validation

- R16 hardware: HLS C simulation and C/RTL cosimulation passed; Vivado routed
  with positive setup and hold slack. The same 96 convolution intermediates and
  three final outputs matched R15 bit-for-bit on the board.
- R17 concatenation fusion: exact but no whole-graph speedup; rejected.
- R18 preview annotation cadence (25 ms to 120 ms): slight live improvement
  within measurement noise; rejected as a production change.
- R19 XRT default BO cache-sync removal: exact for 96 convolution outputs,
  three graph outputs, and benchmark input variants. The XRT implementation on
  this image allocates BOs with `cacheable=0` by default; this is intentionally
  board-specific and must be revisited if PYNQ/XRT or allocation flags change.

With the R16 bitstream, nine same-input graph runs averaged 0.4080 s on R15,
0.3850 s skipping redundant output sync, and 0.3732 s skipping both redundant
input/output sync. The PL kernel still takes about 0.184 s of that 0.373 s.
In the live service A/B test, R15 counted 1.577 pose inferences/s and R19
counted 1.660/s. After deployment a second R19 run counted 1.657/s with camera,
preview, and electrode model configured and no service error. No person was in
the camera for these measurements, so successful electrode recognition remains
untested in this specific run.

## Next architectural step

The 0.373 s graph itself caps true model inference at 2.68 FPS before camera,
JPEG, and electrode work. Reaching 5 FPS requires a graph redesign: execute
several consecutive quantized operators per PL invocation, stream or tile their
intermediates on chip, and reduce PS↔PL synchronization. A focused prototype
should fuse the costly first backbone pair, verify exact activations, measure
DDR bytes and total graph time, and only then expand to more layers. If the
resource/latency budget does not close on XC7Z020, evaluate a smaller pose
architecture trained with quantization-aware training without a teacher model;
judge it against the existing point-placement and electrode datasets, not just
generic keypoint scores. Preview frame rate must never be reported as model FPS.
