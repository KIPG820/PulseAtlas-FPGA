"""Upload isolated cache-sync variants, compare exact outputs, restore service."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import paramiko

ROOT = Path(__file__).resolve().parent
REMOTE = "/home/xilinx/PulseAtlas_B_Vision_R19_UncachedSync"
FILES = {str(path.relative_to(ROOT)).replace("\\", "/"): path
         for variant in ("variant_out", "variant_both")
         for path in (ROOT / variant / "vision_guide").glob("*.py")}
FILES["run_board_comparison.sh"] = ROOT / "run_board_comparison.sh"


def run(client, command, sudo=False):
    if sudo:
        command = "sudo -S -p '' " + command
    stdin, stdout, stderr = client.exec_command(command, timeout=600)
    if sudo:
        stdin.write(os.environ["PULSEATLAS_B_PASSWORD"] + "\n")
        stdin.flush()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("192.168.2.100", username="xilinx",
                   password=os.environ["PULSEATLAS_B_PASSWORD"],
                   look_for_keys=False, allow_agent=False, timeout=10)
    try:
        rc, _, _ = run(client, f"test ! -e {REMOTE}")
        if rc:
            raise RuntimeError(f"refusing to overwrite {REMOTE}")
        rc, _, err = run(client, f"mkdir -p {REMOTE}/variant_out/vision_guide "
                             f"{REMOTE}/variant_both/vision_guide")
        if rc:
            raise RuntimeError(err)
        sftp = client.open_sftp()
        try:
            for relative, source in FILES.items():
                target = REMOTE + "/" + relative
                sftp.put(str(source), target)
                rc, out, err = run(client, f"sha256sum '{target}'")
                if rc or out.split()[0] != hashlib.sha256(source.read_bytes()).hexdigest():
                    raise RuntimeError(f"checksum mismatch {relative}: {err}")
        finally:
            sftp.close()
        rc, out, err = run(client, f"/bin/bash {REMOTE}/run_board_comparison.sh", sudo=True)
        print(out, flush=True)
        if err:
            print(err, file=sys.stderr, flush=True)
        (ROOT / "validation").mkdir(exist_ok=True)
        sftp = client.open_sftp()
        try:
            for name in ("baseline", "out", "both"):
                for ext in ("json", "log"):
                    source = f"/tmp/pulseatlas-r19-results/{name}.{ext}"
                    try:
                        sftp.get(source, str(ROOT / "validation" / f"{name}.{ext}"))
                    except FileNotFoundError:
                        pass
        finally:
            sftp.close()
        return rc
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
