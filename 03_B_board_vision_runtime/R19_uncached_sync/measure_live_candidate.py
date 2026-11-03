"""Temporary R16 bitstream + R19 backend live test with automatic restore."""
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
HOST = "192.168.2.100"
UNIT = "/etc/systemd/system/pulseatlas-b-vision.service"
BACKUP = "/tmp/pulseatlas-b-vision-r19-test.service"
UPLOAD = "/home/xilinx/PulseAtlas_B_Vision_R19_UncachedSync/r19-test.service"
OLD_BIT = "/home/xilinx/PulseAtlas_B_Vision_R14_90MHz/vision_accel/vision_bonly_writebase_95mhz.bit"
NEW_BIT = "/home/xilinx/PulseAtlas_B_Vision_R16_Depthwise/vision_accel/vision_bonly_depthwise_95mhz.bit"
OLD_ROOT = "/home/xilinx/PulseAtlas_B_Vision_R15_ZeroCopy"
NEW_ROOT = "/home/xilinx/PulseAtlas_B_Vision_R19_UncachedSync/variant_both"
MEASURE = ROOT.parent / "PulseAtlas_B_Vision_R7_PreparedGraph" / "tools" / "measure_b_service.py"


def run(client, command, sudo=False):
    if sudo:
        command = "sudo -S -p '' /bin/bash -lc " + shlex.quote(command)
    stdin, stdout, stderr = client.exec_command(command, timeout=90)
    if sudo:
        stdin.write(os.environ["PULSEATLAS_B_PASSWORD"] + "\n")
        stdin.flush()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if rc:
        raise RuntimeError(f"{rc}: {command}\n{out}\n{err}")
    return out


def wait_port():
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, 8765), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("B vision service did not open port")


def measure(name):
    output = ROOT / "validation" / f"live_{name}.json"
    output.parent.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(MEASURE), "--host", HOST,
                    "--seconds", "42", "--warmup", "20", "--output", str(output)],
                   check=True)


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="xilinx", password=os.environ["PULSEATLAS_B_PASSWORD"],
                   look_for_keys=False, allow_agent=False, timeout=10)
    backed_up = False
    try:
        # Only the two Python modules differ; link immutable model/calibration
        # files so the board does not accumulate another large runtime copy.
        run(client, "for source in " + OLD_ROOT + "/vision_guide/*; do "
                    "dest=" + NEW_ROOT + "/vision_guide/$(basename \"$source\"); "
                    "if [ ! -e \"$dest\" ]; then ln -s \"$source\" \"$dest\"; fi; "
                    "done; test -L " + NEW_ROOT + "/vision_accel || "
                    "ln -s " + OLD_ROOT + "/vision_accel " + NEW_ROOT + "/vision_accel")
        sftp = client.open_sftp()
        try:
            original = sftp.open(UNIT, "rb").read().decode("utf-8")
            if original.count(OLD_BIT) != 1 or original.count(OLD_ROOT) != 1:
                raise RuntimeError("unexpected installed service; aborting")
            candidate = original.replace(OLD_BIT, NEW_BIT).replace(OLD_ROOT, NEW_ROOT)
            with sftp.open(UPLOAD, "wb") as stream:
                stream.write(candidate.encode("utf-8"))
        finally:
            sftp.close()
        print("Measuring deployed R15 baseline", flush=True)
        measure("r15")
        run(client, f"test ! -e {BACKUP}")
        run(client, f"cp {UNIT} {BACKUP}", sudo=True)
        backed_up = True
        run(client, f"cp {UPLOAD} {UNIT} && systemctl daemon-reload && "
                    "systemctl restart pulseatlas-b-vision.service", sudo=True)
        wait_port()
        time.sleep(12)
        print("Measuring R16 + R19 candidate", flush=True)
        measure("r19")
    finally:
        if backed_up:
            run(client, f"cp {BACKUP} {UNIT} && systemctl daemon-reload && "
                        "systemctl restart pulseatlas-b-vision.service && "
                        f"rm -f {BACKUP} {UPLOAD}", sudo=True)
            print("Original R15 service restored", flush=True)
        client.close()


if __name__ == "__main__":
    main()
