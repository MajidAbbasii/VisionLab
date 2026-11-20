"""Create a deterministic, held-out COCO test sample with verified labels.

The sample is drawn from COCO train2017 while excluding the project's existing
training and validation images. The selection manifest and checksums preserve
provenance; rerunning the script reuses valid downloads and retries failures.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import random
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path('datasets/coco2017_independent_test')
ANNOTATIONS = Path('datasets/coco2017/annotations/instances_train2017.json')
CLASSES = Path('datasets/classes.txt')
TRAIN_IMAGES = Path(r'F:\datasets\coco128\images\train2017')
SAMPLE_SIZE = 5000
SEED = 20261119
WORKERS = 12

image_dir = ROOT / 'images' / 'test2017'
label_dir = ROOT / 'labels' / 'test2017'
annotation_dir = ROOT / 'annotations'
for directory in (image_dir, label_dir, annotation_dir):
    directory.mkdir(parents=True, exist_ok=True)

print('Loading COCO train2017 annotations...', flush=True)
with ANNOTATIONS.open('r', encoding='utf-8') as f:
    coco = json.load(f)
class_names = [line.strip() for line in CLASSES.read_text(encoding='utf-8').splitlines() if line.strip()]
if len(class_names) != 80:
    raise RuntimeError(f'Expected 80 classes, found {len(class_names)}')

train_names = {p.name for p in TRAIN_IMAGES.glob('*.jpg')}
val_names = {p.name for p in (Path('datasets/coco2017/images/val2017')).glob('*.jpg')}
images_by_name = {item['file_name']: item for item in coco['images']}
missing_train = sorted(train_names - set(images_by_name))
if missing_train:
    raise RuntimeError(f'{len(missing_train)} COCO128 train images missing from COCO train annotations: {missing_train[:5]}')

# Reserve this sample before expanded training/validation selection. Excluding
# both known splits prevents image-level leakage into the independent test set.
eligible = [item for item in coco['images'] if item['file_name'] not in train_names and item['file_name'] not in val_names]
if len(eligible) < SAMPLE_SIZE:
    raise RuntimeError(f'Only {len(eligible)} eligible images; need {SAMPLE_SIZE}')
rng = random.Random(SEED)
selected = sorted(rng.sample(eligible, SAMPLE_SIZE), key=lambda item: int(item['id']))
selected_ids = {int(item['id']) for item in selected}
selected_names = {item['file_name'] for item in selected}
if selected_names & train_names or selected_names & val_names:
    raise RuntimeError('Split overlap detected before download')

print(f'Selected {len(selected)} images; excluding {len(train_names)} COCO128 train images; seed={SEED}', flush=True)

# Persist the exact image IDs and URLs before downloading so an interrupted run
# resumes the same sample instead of selecting a different test population.
manifest = {
    'dataset': 'COCO 2017 train images held out as independent test',
    'split': 'test2017',
    'source_split': 'train2017',
    'selection_seed': SEED,
    'images_requested': SAMPLE_SIZE,
    'excluded_training_images': len(train_names),
    'excluded_existing_validation_images': len(val_names),
    'images': [{'id': int(x['id']), 'file_name': x['file_name'], 'width': int(x['width']), 'height': int(x['height']), 'url': x['coco_url']} for x in selected],
}
(annotation_dir / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def download_one(item):
    """Download and validate one selected JPEG, retrying transient failures."""
    name = item['file_name']
    target = image_dir / name
    if target.is_file() and target.stat().st_size > 1024:
        with target.open('rb') as f:
            if f.read(2) == b'\xff\xd8':
                return name, 'cached', target.stat().st_size
    # COCO's published coco_url uses HTTP. Use the official URL as published;
    # validate the JPEG signature and non-trivial payload before accepting it.
    url = item['coco_url']
    last_error = None
    for attempt in range(1, 6):
        tmp = target.with_suffix('.jpg.part')
        try:
            result = subprocess.run(
                ['curl.exe', '--fail', '--silent', '--show-error', '--location', '--max-time', '50', '--output', str(tmp), url],
                capture_output=True, text=True, timeout=55,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f'curl exit {result.returncode}')
            if not tmp.is_file() or tmp.stat().st_size < 1024:
                raise RuntimeError('Downloaded image is missing or too small')
            with tmp.open('rb') as f:
                if f.read(2) != b'\xff\xd8':
                    raise RuntimeError('Response is not a JPEG image')
            tmp.replace(target)
            return name, 'downloaded', target.stat().st_size
        except Exception as exc:
            last_error = exc
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(min(attempt * 1.0, 5))
    return name, f'FAILED: {type(last_error).__name__}: {last_error}', 0

print(f'Downloading images with {WORKERS} workers (resumable)...', flush=True)
failed = []
downloaded = 0
cached = 0
total_bytes = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = [pool.submit(download_one, item) for item in selected]
    for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
        name, status, size = future.result()
        total_bytes += size
        if status == 'downloaded':
            downloaded += 1
        elif status == 'cached':
            cached += 1
        else:
            failed.append({'file_name': name, 'error': status})
        if index % 250 == 0 or index == SAMPLE_SIZE:
            print(f'Progress {index}/{SAMPLE_SIZE}; downloaded={downloaded}; cached={cached}; failed={len(failed)}; MB={total_bytes/1024/1024:.1f}', flush=True)

if failed:
    (annotation_dir / 'download_failures.json').write_text(json.dumps(failed, indent=2), encoding='utf-8')
    raise RuntimeError(f'{len(failed)} image downloads failed; rerun script to resume')

# Convert only annotations belonging to the selected image IDs. Category IDs
# are translated through class names because COCO IDs are not model class indices.
category_by_name = {item['name']: int(item['id']) for item in coco['categories']}
missing_classes = sorted(set(class_names) - set(category_by_name))
if missing_classes:
    raise RuntimeError(f'Missing categories: {missing_classes}')
class_by_category = {category_by_name[name]: index for index, name in enumerate(class_names)}
selected_by_id = {int(item['id']): item for item in selected}
rows = {image_id: [] for image_id in selected_ids}
class_counts = Counter()
skipped_crowd = skipped_invalid = 0
for ann in coco['annotations']:
    image_id = int(ann['image_id'])
    if image_id not in selected_ids:
        continue
    if int(ann.get('iscrowd', 0)) == 1:
        skipped_crowd += 1
        continue
    category_id = int(ann['category_id'])
    if category_id not in class_by_category:
        skipped_invalid += 1
        continue
    x, y, width, height = (float(v) for v in ann['bbox'])
    if width <= 0 or height <= 0:
        skipped_invalid += 1
        continue
    img = selected_by_id[image_id]
    iw, ih = float(img['width']), float(img['height'])
    cx, cy, nw, nh = (x + width / 2) / iw, (y + height / 2) / ih, width / iw, height / ih
    if not all(0 <= v <= 1 for v in (cx, cy, nw, nh)):
        skipped_invalid += 1
        continue
    cls = class_by_category[category_id]
    rows[image_id].append(f'{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
    class_counts[cls] += 1


for image in selected:
    image_id = int(image['id'])
    label_path = label_dir / (Path(image['file_name']).stem + '.txt')
    label_path.write_text('\n'.join(rows[image_id]) + ('\n' if rows[image_id] else ''), encoding='utf-8')

selected_annotations = [ann for ann in coco['annotations'] if int(ann['image_id']) in selected_ids]
subset = {
    'info': coco.get('info', {}),
    'licenses': coco.get('licenses', []),
    'images': selected,
    'annotations': selected_annotations,
    'categories': coco['categories'],
}
(annotation_dir / 'instances_test2017.json').write_text(json.dumps(subset, separators=(',', ':')), encoding='utf-8')

names_yaml = '\n'.join(f'  {index}: {name}' for index, name in enumerate(class_names))
yaml = (
    '# Held-out COCO2017 train2017 sample. Do not use for training, tuning, or threshold selection.\n'
    'path: F:/Projects/VisionLab/datasets/coco2017_independent_test\n'
    'train: images/test2017\n'
    'val: images/test2017\n'
    'names:\n' + names_yaml + '\n'
)
Path('datasets/coco2017_independent_test.yaml').write_text(yaml, encoding='utf-8')

# Verify image signatures, label syntax/ranges, and split separation; then
# produce per-image SHA-256 checksums to identify the exact downloaded sample.
missing_images = []
missing_labels = []
invalid_labels = []
sha_rows = []
for image in selected:
    name = image['file_name']
    ip = image_dir / name
    lp = label_dir / (Path(name).stem + '.txt')
    if not ip.is_file() or ip.stat().st_size < 1024:
        missing_images.append(name)
        continue
    if not lp.is_file():
        missing_labels.append(name)
        continue
    raw = ip.read_bytes()
    if raw[:2] != b'\xff\xd8':
        invalid_labels.append(f'{name}: invalid JPEG signature')
    sha_rows.append({'file_name': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    for line_no, line in enumerate(lp.read_text(encoding='utf-8').splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            invalid_labels.append(f'{lp.name}:{line_no}: expected 5 fields')
            continue
        cls = int(parts[0])
        vals = [float(x) for x in parts[1:]]
        if not 0 <= cls < 80 or not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            invalid_labels.append(f'{lp.name}:{line_no}: out-of-range label')

if missing_images or missing_labels or invalid_labels:
    raise RuntimeError(json.dumps({'missing_images': missing_images[:10], 'missing_labels': missing_labels[:10], 'invalid_labels': invalid_labels[:10]}, indent=2))
if selected_names & train_names or selected_names & val_names:
    raise RuntimeError('Final split overlap detected')

(annotation_dir / 'sha256_manifest.json').write_text(json.dumps(sha_rows, indent=2), encoding='utf-8')
summary = {
    'status': 'complete',
    'images': len(selected),
    'label_files': len(list(label_dir.glob('*.txt'))),
    'annotations': len(selected_annotations),
    'yolo_boxes_written': sum(class_counts.values()),
    'skipped_crowd': skipped_crowd,
    'skipped_invalid': skipped_invalid,
    'classes_with_annotations': len(class_counts),
    'class_counts': {class_names[index]: count for index, count in sorted(class_counts.items())},
    'training_overlap': 0,
    'existing_validation_overlap': 0,
    'downloaded_this_run': downloaded,
    'cached': cached,
    'total_image_bytes': total_bytes,
    'seed': SEED,
    'dataset_yaml': 'datasets/coco2017_independent_test.yaml',
    'annotations_json': str(annotation_dir / 'instances_test2017.json'),
    'sha256_manifest': str(annotation_dir / 'sha256_manifest.json'),
}
(annotation_dir / 'preparation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2), flush=True)




