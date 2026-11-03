#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/home/xilinx/pulseatlas_a_r14_ui_20261027
PACKAGE=/tmp/PulseAtlas_A_Board_R15_TFT_Filter_20261027.zip
BACKUP=/tmp/pulseatlas_a_r14_pre_r15_tft_filter_20261027.tar.gz
LOG=/home/xilinx/pulseatlas_r15_tft_filter.log
STAGE=$(mktemp -d /tmp/pulseatlas_r15_tft_filter.XXXXXX)
cleanup() { rm -rf -- "$STAGE"; }
trap cleanup EXIT

[ -f "$PACKAGE" ]
[ -d "$APP_DIR" ]
unzip -tq "$PACKAGE"
unzip -q "$PACKAGE" -d "$STAGE"
(cd "$STAGE" && sha256sum -c SHA256SUMS.txt >/tmp/pulseatlas_r15_package_verify.txt)

# Preserve the live UI source and manifest for an immediate rollback.
tar -czf "$BACKUP" -C "$APP_DIR" pulseatlas_r9_app.py pulseatlas_tft_render.py SHA256SUMS.txt

# Stop only the app's Python process; the R9 service is inactive on this board.
mapfile -t PIDS < <(pgrep -f '^/usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py$' || true)
for pid in "${PIDS[@]}"; do kill -TERM "$pid"; done
for attempt in $(seq 1 5); do
  if ! pgrep -f '^/usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py$' >/dev/null; then break; fi
  sleep 1
done
mapfile -t LEFTOVER_PIDS < <(pgrep -f '^/usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py$' || true)
for pid in "${LEFTOVER_PIDS[@]}"; do kill -KILL "$pid"; done
sleep 1
if pgrep -af '^/usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py$'; then
  echo "ERROR: app process is being restarted externally; package was not installed." >&2
  exit 1
fi

unzip -oq "$PACKAGE" -d "$APP_DIR"
(cd "$APP_DIR" && sha256sum -c SHA256SUMS.txt >/tmp/pulseatlas_r15_installed_verify.txt)
sha256sum "$APP_DIR/pulseatlas_tft_render.py" "$APP_DIR/pulseatlas_r9_app.py"

cd "$APP_DIR"
nohup /usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py >"$LOG" 2>&1 </dev/null &
for attempt in $(seq 1 12); do
  if pgrep -f '^/usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py$' >/tmp/pulseatlas_r15_pids.txt; then break; fi
  sleep 1
done
if ! pgrep -f '^/usr/local/share/pynq-venv/bin/python run_pulseatlas_r9.py$' >/tmp/pulseatlas_r15_pids.txt; then
  echo "ERROR: R15 app failed to stay running; startup log follows." >&2
  tail -n 80 "$LOG" >&2
  exit 2
fi
printf 'R15 app PID(s): '
tr '\n' ' ' </tmp/pulseatlas_r15_pids.txt
printf '\nInstalled package manifest: PASS\n'
printf 'Backup: %s\nLog: %s\n' "$BACKUP" "$LOG"
tail -n 30 "$LOG"
rm -f -- "$PACKAGE" /tmp/pulseatlas_r15_package_verify.txt /tmp/pulseatlas_r15_installed_verify.txt /tmp/pulseatlas_r15_pids.txt
