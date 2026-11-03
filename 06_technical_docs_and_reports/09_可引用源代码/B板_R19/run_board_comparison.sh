#!/bin/bash
set -euo pipefail
. /etc/profile.d/xrt_setup.sh 2>/dev/null || true
ROOT=/home/xilinx/PulseAtlas_B_Vision_R7_20261013
R15=/home/xilinx/PulseAtlas_B_Vision_R15_ZeroCopy/vision_guide
R19=/home/xilinx/PulseAtlas_B_Vision_R19_UncachedSync
BENCHMARK=/home/xilinx/PulseAtlas_B_Vision_R16_Depthwise/benchmark_int8_graph_board.py
OLD_BIT=/home/xilinx/PulseAtlas_B_Vision_R14_90MHz/vision_accel/vision_bonly_writebase_95mhz.bit
TEST_BIT=/home/xilinx/PulseAtlas_B_Vision_R16_Depthwise/vision_accel/vision_bonly_depthwise_95mhz.bit
PY=/usr/local/share/pynq-venv/bin/python
RESULTS=/tmp/pulseatlas-r19-results
test -f "$TEST_BIT" && test -f "${TEST_BIT%.bit}.hwh"
was_active=0
if systemctl is-active --quiet pulseatlas-b-vision.service; then was_active=1; fi
restore() {
    rc=$?
    trap - EXIT HUP INT TERM
    "$PY" - "$OLD_BIT" <<'RESTORE' || rc=1
import sys
from pynq import Overlay
Overlay(sys.argv[1], download=True)
RESTORE
    if [ "$was_active" = 1 ]; then
        systemctl start pulseatlas-b-vision.service || rc=1
    fi
    exit "$rc"
}
trap restore EXIT
trap 'exit 130' HUP INT TERM
systemctl stop pulseatlas-b-vision.service
mkdir -p "$RESULTS"
for name in baseline out both; do
    candidate="$R15"
    if [ "$name" = out ]; then candidate="$R19/variant_out/vision_guide"; fi
    if [ "$name" = both ]; then candidate="$R19/variant_both/vision_guide"; fi
    timeout 180s "$PY" "$BENCHMARK" --backend r7-pool \
        --r3-root "$ROOT" --candidate-python "$candidate" \
        --bit "$TEST_BIT" --runs 9 --output "$RESULTS/$name.json" \
        >"$RESULTS/$name.log" 2>&1 || {
            tail -35 "$RESULTS/$name.log"
            exit 1
        }
done
"$PY" - "$RESULTS" <<'COMPARE'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); a=json.loads((p/'baseline.json').read_text())
for name in ('out', 'both'):
    b=json.loads((p/f'{name}.json').read_text())
    assert a['all_intermediates_and_outputs_exact'] and b['all_intermediates_and_outputs_exact']
    assert a['variant_fingerprints_excluded'] == b['variant_fingerprints_excluded']
    print(name, 'ALL_96_CONVS_AND_3_OUTPUTS_EXACT')
    print(name, 'graph_s', b['mean_s'], 'speedup', a['mean_s']/b['mean_s'])
    print(name, 'backend', b['mean_backend_s_per_run'])
print('baseline_graph_s', a['mean_s'])
print('baseline_backend', a['mean_backend_s_per_run'])
COMPARE
