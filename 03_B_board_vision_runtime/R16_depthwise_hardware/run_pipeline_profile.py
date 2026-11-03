"""Run the B-board stage profiler while preserving the current service."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import paramiko


ROOT = Path(__file__).resolve().parent
REMOTE = "/home/xilinx/PulseAtlas_B_Vision_R16_Depthwise"


def main() -> int:
    password = os.environ.get("PULSEATLAS_B_PASSWORD")
    if not password:
        raise SystemExit("Set PULSEATLAS_B_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("192.168.2.100", username="xilinx", password=password,
                   look_for_keys=False, allow_agent=False, timeout=10)
    try:
        sftp = client.open_sftp()
        try:
            for name in ("profile_live_pipeline_board.py", "profile_live_pipeline.sh"):
                source = ROOT / name
                target = REMOTE + "/" + name
                sftp.put(str(source), target)
                with sftp.open(target, "rb") as file:
                    if hashlib.sha256(file.read()).digest() != hashlib.sha256(source.read_bytes()).digest():
                        raise RuntimeError(f"checksum mismatch: {name}")
        finally:
            sftp.close()
        print("pipeline profiler uploaded", flush=True)
        command = f"sudo -S -p '' /bin/bash {REMOTE}/profile_live_pipeline.sh"
        stdin, stdout, stderr = client.exec_command(command, timeout=240)
        stdin.write(password + "\n")
        stdin.flush()
        output = stdout.read().decode("utf-8", "replace")
        error = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        print(output, flush=True)
        if error:
            print(error, file=sys.stderr, flush=True)
        print("pipeline_profile_exit", code, flush=True)
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
