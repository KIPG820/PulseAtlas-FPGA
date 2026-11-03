"""Audit the supplied video-disjoint, same-subject ten-class dataset without changing it."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import yaml

NAMES = ['RA', 'LA', 'RL', 'LL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(package, output):
    import cv2
    package, output = Path(package).resolve(), Path(output).resolve()
    root = package / 'datasets/electrode_detector'
    cfg = yaml.safe_load((root/'data.yaml').read_text(encoding='utf-8'))
    if cfg['names'] != dict(enumerate(NAMES)):
        raise ValueError('Expected ten electrode-pad classes in the provided order')
    with (root/'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_image, groups, splits = {}, {}, {}
    errors, digest_splits, fingerprint = [], {}, hashlib.sha256()
    image_sizes, short_sides = Counter(), defaultdict(list)
    for row in rows:
        key, split = row['image'].replace('\\','/'), row['split']
        if key in by_image:
            errors.append(f'Duplicate manifest image: {key}')
        by_image[key] = row
        video = row['source_video']
        if video in groups and groups[video] != split:
            errors.append(f'Video leakage: {video}')
        groups[video] = split
    for split in ('train', 'val', 'test'):
        images = sorted((root/cfg[split]).glob('*.jpg'))
        counts, backgrounds = Counter(), 0
        if not images:
            errors.append(f'Empty {split}')
        for image in images:
            rel = image.relative_to(root).as_posix()
            label = root/'labels'/split/(image.stem+'.txt')
            row = by_image.get(rel)
            if not row or row['split'] != split or row['label'].replace('\\','/') != label.relative_to(root).as_posix():
                errors.append(f'Manifest mismatch: {rel}')
            if not label.exists():
                errors.append(f'Missing label: {label}')
                continue
            decoded = cv2.imread(str(image))
            if decoded is None:
                errors.append(f'Image cannot decode: {rel}')
                continue
            height, width = decoded.shape[:2]
            image_sizes[f'{width}x{height}'] += 1
            digest = sha256(image)
            if digest in digest_splits and digest_splits[digest] != split:
                errors.append(f'Exact image leakage: {rel}')
            digest_splits[digest] = split
            fingerprint.update((rel+digest+sha256(label)).encode())
            lines = label.read_text(encoding='utf-8').splitlines()
            backgrounds += not bool(lines)
            for n, line in enumerate(lines, 1):
                try:
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError()
                    cls = int(fields[0])
                    x,y,w,h = map(float, fields[1:])
                    valid = 0 <= cls < 10 and all(math.isfinite(v) for v in (x,y,w,h)) and 0 < w <= 1 and 0 < h <= 1
                    valid = valid and x-w/2 >= -1e-6 and x+w/2 <= 1+1e-6 and y-h/2 >= -1e-6 and y+h/2 <= 1+1e-6
                    if not valid:
                        raise ValueError()
                    counts[NAMES[cls]] += 1
                    if split == 'train':
                        short_sides[NAMES[cls]].append(min(w*width,h*height)*960/max(width,height))
                except ValueError:
                    errors.append(f'Invalid box: {label}:{n}')
        splits[split] = dict(images=len(images), boxes=dict(counts), background_images=backgrounds,
            absent_classes=[name for name in NAMES if not counts[name]])
        labels = list((root/'labels'/split).glob('*.txt'))
        if {p.stem for p in images} != {p.stem for p in labels}:
            errors.append(f'Orphan/missing labels in {split}')
    if len(rows) != sum(s['images'] for s in splits.values()):
        errors.append('Manifest/image count mismatch')
    if splits['train']['absent_classes']:
        errors.append('Training does not contain all ten classes')
    poses = {p.name:sha256(p) for p in (package/'models/pose').iterdir() if p.is_file()}
    expected_pose = json.loads((package/'models/pose/yolo11n-pose-manifest.json').read_text())['sha256']
    if poses['yolo11n-pose.pt'] != expected_pose:
        errors.append('Pose PT hash differs from provided manifest')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(package=str(package), dataset_root=str(root), names=NAMES, splits=splits,
        source_video_splits=groups, dimensions=dict(image_sizes), fingerprint=fingerprint.hexdigest(),
        pose_sha256=poses, errors=errors,
        limitations=['Same subject/session; no independent subject generalization',
            'Val only RL/LL; test only V3/V5; remaining classes not independently measured',
            'No confirmed negative images; annotation identities not independently re-reviewed'],
        train_box_short_side_at_960={k:dict(min=min(v),median=sorted(v)[len(v)//2]) for k,v in short_sides.items()})
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cfg['path'] = str(root)
    (output/'data.resolved.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    if errors:
        raise RuntimeError('\n'.join(errors[:30]))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--package',required=True)
    p.add_argument('--output',default='artifacts/pulseatlas')
    a=p.parse_args()
    audit(a.package,a.output)
