"""Upload the isolated R16 bitstream and compare it with the deployed R15 runtime.

Set PULSEATLAS_B_PASSWORD in the environment. The on-board shell script always
restores the R15 service and R14 bitstream, including after a failed comparison.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import paramiko


ROOT = Path(__file__).resolve().parent
REMOTE = "/home/xilinx/PulseAtlas_B_Vision_R16_Depthwise"
FILES = {
    "vision_accel/vision_bonly_depthwise_95mhz.bit": ROOT / "vision_accel/vision_bonly_depthwise_95mhz.bit",
    "vision_accel/vision_bonly_depthwise_95mhz.hwh": ROOT / "vision_accel/vision_bonly_depthwise_95mhz.hwh",
    "benchmark_int8_graph_board.py": ROOT.parent / "PulseAtlas_B_Vision_R15_ZeroCopy/tools/benchmark_int8_graph_board.py",
    "run_board_comparison.sh": ROOT / "run_board_comparison.sh",
}


def run(client: paramiko.SSHClient, command: str, *, sudo: bool = False):
    if sudo:
        command = "sudo -S -p '' " + command
    stdin, stdout, stderr = client.exec_command(command, timeout=600)
    if sudo:
        stdin.write(os.environ["PULSEATLAS_B_PASSWORD"] + "\n")
        stdin.flush()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    password = os.environ.get("PULSEATLAS_B_PASSWORD")
    if not password:
        raise SystemExit("Set PULSEATLAS_B_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("192.168.2.100", username="xilinx", password=password,
                   look_for_keys=False, allow_agent=False, timeout=10)
    try:
        rc, out, err = run(client, f"test ! -e {REMOTE}")
        if rc:
            raise RuntimeError(f"Refusing to overwrite existing {REMOTE}")
        rc, out, err = run(client, f"mkdir -p {REMOTE}/vision_accel")
        if rc:
            raise RuntimeError(err)
        sftp = client.open_sftp()
        try:
            for relative, source in FILES.items():
                target = REMOTE + "/" + relative
                sftp.put(str(source), target)
                rc, out, err = run(client, f"sha256sum '{target}'")
                if rc or out.split()[0] != hashlib.sha256(source.read_bytes()).hexdigest():
                    raise RuntimeError(f"Upload checksum mismatch: {relative}")
                print("verified", relative, flush=True)
        finally:
            sftp.close()
        rc, out, err = run(client, f"/bin/bash {REMOTE}/run_board_comparison.sh", sudo=True)
        print(out, flush=True)
        if err:
            print(err, file=sys.stderr, flush=True)
        print("comparison_exit", rc, flush=True)
        (ROOT / "validation").mkdir(exist_ok=True)
        sftp = client.open_sftp()
        try:
            for name in ("r15", "r16"):
                for extension in ("json", "log"):
                    source = f"/tmp/pulseatlas-r16-results/{name}.{extension}"
                    target = ROOT / "validation" / f"board_{name}.{extension}"
                    try:
                        sftp.get(source, str(target))
                        print("saved", target, flush=True)
                    except FileNotFoundError:
                        pass
        finally:
            sftp.close()
        return rc
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
