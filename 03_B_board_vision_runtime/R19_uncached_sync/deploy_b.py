"""Promote the validated R16+R19 candidate with a saved rollback unit."""
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
ROLLBACK = "/home/xilinx/PulseAtlas_B_Vision_R19_UncachedSync/service_before_r19.service"
STAGED = "/home/xilinx/PulseAtlas_B_Vision_R19_UncachedSync/service_r19.service"
OLD_BIT = "/home/xilinx/PulseAtlas_B_Vision_R14_90MHz/vision_accel/vision_bonly_writebase_95mhz.bit"
NEW_BIT = "/home/xilinx/PulseAtlas_B_Vision_R16_Depthwise/vision_accel/vision_bonly_depthwise_95mhz.bit"
OLD_ROOT = "/home/xilinx/PulseAtlas_B_Vision_R15_ZeroCopy"
NEW_ROOT = "/home/xilinx/PulseAtlas_B_Vision_R19_UncachedSync/variant_both"
MEASURE = ROOT.parent / "PulseAtlas_B_Vision_R7_PreparedGraph" / "tools" / "measure_b_service.py"


def run(client, cmd, sudo=False):
    if sudo:
        cmd = "sudo -S -p '' /bin/bash -lc " + shlex.quote(cmd)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=90)
    if sudo:
        stdin.write(os.environ["PULSEATLAS_B_PASSWORD"] + "\n")
        stdin.flush()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if rc:
        raise RuntimeError(f"{rc}: {cmd}\n{out}\n{err}")
    return out


def wait_port():
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, 8765), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("service did not open port 8765")


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="xilinx", password=os.environ["PULSEATLAS_B_PASSWORD"],
                   look_for_keys=False, allow_agent=False, timeout=10)
    rollback_written = False
    try:
        sftp = client.open_sftp()
        try:
            current = sftp.open(UNIT, "rb").read().decode("utf-8")
            if current.count(OLD_BIT) != 1 or current.count(OLD_ROOT) != 1:
                raise RuntimeError("installed unit differs from tested R15 baseline")
            if sftp.stat(NEW_BIT).st_size < 1_000_000:
                raise RuntimeError("R16 bitstream missing or incomplete")
            sftp.stat(NEW_BIT.replace(".bit", ".hwh"))
            for path in (NEW_ROOT + "/vision_guide/vision_int8_graph.py",
                         NEW_ROOT + "/vision_guide/vision_int8_prepared_pl.py",
                         NEW_ROOT + "/vision_accel/golden_frame8.npz"):
                sftp.stat(path)
            candidate = current.replace(OLD_BIT, NEW_BIT).replace(OLD_ROOT, NEW_ROOT)
            candidate = candidate.replace("R15 pose", "R19 pose")
            with sftp.open(STAGED, "wb") as stream:
                stream.write(candidate.encode("utf-8"))
        finally:
            sftp.close()
        run(client, f"test ! -e {ROLLBACK}")
        run(client, f"cp {UNIT} {ROLLBACK}", sudo=True)
        rollback_written = True
        run(client, f"cp {STAGED} {UNIT} && systemctl daemon-reload && "
                    "systemctl restart pulseatlas-b-vision.service", sudo=True)
        wait_port()
        time.sleep(12)
        output = ROOT / "validation" / "live_r19_deployed.json"
        subprocess.run([sys.executable, str(MEASURE), "--host", HOST,
                        "--seconds", "42", "--warmup", "20", "--output", str(output)],
                       check=True)
        import json
        summary = json.loads(output.read_text(encoding="utf-8"))["summary"]
        if (summary["inference_count_delta"] < 15 or summary["last_error"] is not None or
                summary["preview_packets"] < 30):
            raise RuntimeError(f"deployed candidate failed live check: {summary}")
        print("R16 + R19 deployed and live-checked", flush=True)
        print(run(client, "systemctl is-active pulseatlas-b-vision.service"), flush=True)
    except Exception:
        if rollback_written:
            run(client, f"cp {ROLLBACK} {UNIT} && systemctl daemon-reload && "
                        "systemctl restart pulseatlas-b-vision.service", sudo=True)
            print("R15 restored after failed deployment", file=sys.stderr, flush=True)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
