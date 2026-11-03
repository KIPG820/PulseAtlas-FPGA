from pathlib import Path
import hashlib
root=Path(__file__).resolve().parent
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
    expected,name=line.split('  ',1)
    path=(root/name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('FAILED: '+name)
print('All package checksums passed')
