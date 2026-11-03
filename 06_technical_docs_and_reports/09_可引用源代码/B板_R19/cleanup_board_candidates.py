"""Remove only rejected B-board test candidates and temporary benchmark data."""
import os
import shlex

import paramiko

paths = (
    "/home/xilinx/PulseAtlas_B_Vision_R17_OperatorFusion",
    "/home/xilinx/PulseAtlas_B_Vision_R18_PreviewCadence",
    "/tmp/pulseatlas-r16-results",
    "/tmp/pulseatlas-r17-results",
    "/tmp/pulseatlas-r19-results",
)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.2.100", username="xilinx",
               password=os.environ["PULSEATLAS_B_PASSWORD"],
               look_for_keys=False, allow_agent=False, timeout=10)
try:
    command = r"""set -eu
unit=$(systemctl cat pulseatlas-b-vision.service)
for p in "$@"; do
    case "$p" in
      /home/xilinx/PulseAtlas_B_Vision_R17_OperatorFusion|\
      /home/xilinx/PulseAtlas_B_Vision_R18_PreviewCadence|\
      /tmp/pulseatlas-r16-results|/tmp/pulseatlas-r17-results|/tmp/pulseatlas-r19-results) ;;
      *) exit 2 ;;
    esac
    if printf '%s' "$unit" | grep -Fq "$p"; then exit 3; fi
    if [ -e "$p" ]; then
        test ! -L "$p"
        test "$(realpath -e "$p")" = "$p"
        rm -rf -- "$p"
        echo "removed $p"
    fi
done
"""
    shell = "sudo -S -p '' /bin/bash -c " + shlex.quote(command) + " -- " + " ".join(map(shlex.quote, paths))
    stdin, stdout, stderr = client.exec_command(shell)
    stdin.write(os.environ["PULSEATLAS_B_PASSWORD"] + "\n")
    stdin.flush()
    print(stdout.read().decode(), end="")
    error = stderr.read().decode()
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"cleanup exit {status}: {error}")
finally:
    client.close()
