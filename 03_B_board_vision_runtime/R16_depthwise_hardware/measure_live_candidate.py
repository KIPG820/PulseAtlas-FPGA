"""Temporarily run the R15 service with the R16 bitstream, then restore it."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

import paramiko


ROOT = Path(__file__).resolve().parent
UNIT = "/etc/systemd/system/pulseatlas-b-vision.service"
BACKUP = "/tmp/pulseatlas-b-vision-r16-test.service"
UPLOAD = "/home/xilinx/PulseAtlas_B_Vision_R16_Depthwise/r16-test.service"
OLD_BIT = "/home/xilinx/PulseAtlas_B_Vision_R14_90MHz/vision_accel/vision_bonly_writebase_95mhz.bit"
NEW_BIT = "/home/xilinx/PulseAtlas_B_Vision_R16_Depthwise/vision_accel/vision_bonly_depthwise_95mhz.bit"


def run(client: paramiko.SSHClient, command: str, sudo: bool = False):
    if sudo:
        command = "sudo -S -p '' /bin/bash -lc " + shlex.quote(command)
    stdin, stdout, stderr = client.exec_command(command, timeout=60)
    if sudo:
        stdin.write(os.environ["PULSEATLAS_B_PASSWORD"] + "\n")
        stdin.flush()
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"command failed ({code}): {command}\n{output}\n{error}")
    return output


def wait_port(seconds: float = 40.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("192.168.2.100", 8765), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("vision service did not open port 8765")


def main() -> int:
    if not os.environ.get("PULSEATLAS_B_PASSWORD"):
        raise SystemExit("Set PULSEATLAS_B_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("192.168.2.100", username="xilinx",
                   password=os.environ["PULSEATLAS_B_PASSWORD"],
                   look_for_keys=False, allow_agent=False, timeout=10)
    backup_created = False
    try:
        sftp = client.open_sftp()
        try:
            with sftp.open(UNIT, "r") as source:
                original = source.read().decode("utf-8")
            if original.count(OLD_BIT) != 1:
                raise RuntimeError("expected exactly one old bitstream path in service")
            with sftp.open(UPLOAD, "w") as destination:
                destination.write(original.replace(OLD_BIT, NEW_BIT).encode("utf-8"))
        finally:
            sftp.close()
        run(client, f"test ! -e {BACKUP}")
        run(client, f"cp {UNIT} {BACKUP}", sudo=True)
        backup_created = True
        run(client, f"cp {UPLOAD} {UNIT} && systemctl daemon-reload && "
                    "systemctl restart pulseatlas-b-vision.service", sudo=True)
        wait_port()
        print("R16 service ready", flush=True)
        time.sleep(12)
        output = ROOT / "validation" / "live_r16.json"
        output.parent.mkdir(exist_ok=True)
        result = subprocess.run([
            sys.executable,
            str(ROOT.parent / "PulseAtlas_B_Vision_R7_PreparedGraph/tools/measure_b_service.py"),
            "--host", "192.168.2.100", "--seconds", "42", "--warmup", "20",
            "--output", str(output),
        ], check=False)
        return result.returncode
    finally:
        if backup_created:
            try:
                run(client, f"cp {BACKUP} {UNIT} && systemctl daemon-reload && "
                            f"systemctl restart pulseatlas-b-vision.service && "
                            f"rm -f {BACKUP} {UPLOAD}", sudo=True)
                print("R15 service restored", flush=True)
            except Exception as exc:
                print(f"RESTORE FAILED: {exc}", file=sys.stderr, flush=True)
                raise
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
