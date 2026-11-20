from __future__ import annotations

import concurrent.futures
import hashlib
import json
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path('datasets/coco2017_expanded')
COCO_ANNOTATIONS = Path('datasets/coco2017/annotations/instances_train2017.json')
CLASSES_FILE = Path('datasets/classes.txt')
EXISTING_TRAIN_IMAGES = Path(r'F:\datasets\coco128\images\train2017')
EXISTING_TRAIN_LABELS = Path(r'F:\datasets\coco128\labels\train2017')
HELD_OUT_MANIFEST = Path('datasets/coco2017_independent_test/annotations/selection_manifest.json')
TRAIN_TOTAL = 20_000  # Includes the existing 128 COCO images.
VAL_TOTAL = 5_000
SEED = 20261120
WORKERS = 12
MAX_IN_FLIGHT = WORKERS * 4

TRAIN_IMAGE_DIR = ROOT / 'images' / 'train2017'
VAL_IMAGE_DIR = ROOT / 'images' / 'val2017'
TRAIN_LABEL_DIR = ROOT / 'labels' / 'train2017'
VAL_LABEL_DIR = ROOT / 'labels' / 'val2017'
ANNOTATION_DIR = ROOT / 'annotations'

for directory in (TRAIN_IMAGE_DIR, VAL_IMAGE_DIR, TRAIN_LABEL_DIR, VAL_LABEL_DIR, ANNOTATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if not COCO_ANNOTATIONS.is_file() or not HELD_OUT_MANIFEST.is_file():
    raise FileNotFoundError('COCO train annotations or independent-test selection manifest is missing')

class_names = [x.strip() for x in CLASSES_FILE.read_text(encoding='utf-8').splitlines() if x.strip()]
if len(class_names) != 80:
    raise RuntimeError(f'Expected 80 class names, found {len(class_names)}')

existing_train = {p.name for p in EXISTING_TRAIN_IMAGES.glob('*.jpg')}
if len(existing_train) != 128:
    raise RuntimeError(f'Expected exactly 128 existing COCO128 train images, found {len(existing_train)}')

held_out = json.loads(HELD_OUT_MANIFEST.read_text(encoding='utf-8'))
test_names = {x['file_name'] for x in held_out['images']}
if len(test_names) != 5000:
    raise RuntimeError(f'Expected 5000 held-out test images, found {len(test_names)}')

print('Loading COCO train2017 annotations (about 470 MB)...', flush=True)
with COCO_ANNOTATIONS.open('r', encoding='utf-8') as f:
    coco = json.load(f)
images = {x['file_name']: x for x in coco['images']}
missing_existing = sorted(existing_train - set(images))
missing_test = sorted(test_names - set(images))
if missing_existing or missing_test:
    raise RuntimeError(f'Image IDs missing from COCO annotations: train={missing_existing[:5]}, test={missing_test[:5]}')

# Exclude the previously-used official validation set as well as the untouched test set.
official_val_names = {p.name for p in Path('datasets/coco2017/images/val2017').glob('*.jpg')}
if len(official_val_names) != 5000:
    raise RuntimeError(f'Expected 5000 existing official val2017 images, found {len(official_val_names)}')
excluded = existing_train | test_names | official_val_names
eligible = [x for x in coco['images'] if x['file_name'] not in excluded]
additional_train_count = TRAIN_TOTAL - len(existing_train)
needed = additional_train_count + VAL_TOTAL
if len(eligible) < needed:
    raise RuntimeError(f'Only {len(eligible)} eligible images; need {needed}')

rng = random.Random(SEED)
sample = rng.sample(eligible, needed)
sample.sort(key=lambda x: int(x['id']))
train_selected = sample[:additional_train_count]
val_selected = sample[additional_train_count:]
train_selected = sorted([images[n] for n in existing_train] + train_selected, key=lambda x: int(x['id']))
val_selected = sorted(val_selected, key=lambda x: int(x['id']))
train_names = {x['file_name'] for x in train_selected}
val_names = {x['file_name'] for x in val_selected}
if len(train_names) != TRAIN_TOTAL or len(val_names) != VAL_TOTAL:
    raise RuntimeError('Unexpected split size or duplicate filenames')
if train_names & val_names or train_names & test_names or val_names & test_names:
    raise RuntimeError('Train/validation/test overlap detected before download')
if (train_names | val_names) & official_val_names:
    raise RuntimeError('Previously-used official val2017 images must not enter new splits')

def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

selection = {
    'dataset': 'COCO 2017 expanded training and fresh validation',
    'source_split': 'train2017',
    'selection_seed': SEED,
    'train_images_total_including_existing_128': TRAIN_TOTAL,
    'train_images_newly_selected': additional_train_count,
    'validation_images': VAL_TOTAL,
    'excluded_existing_train_images': len(existing_train),
    'excluded_independent_test_images': len(test_names),
    'excluded_previously_used_official_val_images': len(official_val_names),
    'train': [{'id': int(x['id']), 'file_name': x['file_name'], 'width': int(x['width']), 'height': int(x['height']), 'url': x['coco_url']} for x in train_selected],
    'val': [{'id': int(x['id']), 'file_name': x['file_name'], 'width': int(x['width']), 'height': int(x['height']), 'url': x['coco_url']} for x in val_selected],
}
selection_path = ANNOTATION_DIR / 'selection_manifest.json'
if selection_path.exists():
    prior = json.loads(selection_path.read_text(encoding='utf-8'))
    if prior.get('selection_seed') != SEED or prior.get('train_images_total_including_existing_128') != TRAIN_TOTAL or prior.get('validation_images') != VAL_TOTAL:
        raise RuntimeError('An existing selection manifest has different parameters; refusing to overwrite it')
    selection = prior
    train_selected = selection['train']
    val_selected = selection['val']
    train_names = {x['file_name'] for x in train_selected}
    val_names = {x['file_name'] for x in val_selected}
    if train_names & val_names or train_names & test_names or val_names & test_names:
        raise RuntimeError('Existing selection manifest contains split overlap')
else:
    write_json(selection_path, selection)

print(f'Selected {len(train_selected)} train + {len(val_selected)} fresh validation images; test remains excluded.', flush=True)

# Reuse the original 128 image files. Their YOLO labels are regenerated from
# the official COCO annotations below, so no cached labels are required.
for name in existing_train:
    src_img, dst_img = EXISTING_TRAIN_IMAGES / name, TRAIN_IMAGE_DIR / name
    if not dst_img.exists():
        shutil.copy2(src_img, dst_img)

def download_one(item: dict, image_dir: Path) -> tuple[str, str, int]:
    name = item['file_name']
    target = image_dir / name
    if target.is_file() and target.stat().st_size > 1024:
        with target.open('rb') as f:
            if f.read(2) == b'\xff\xd8':
                return name, 'cached', target.stat().st_size
    last_error = None
    for attempt in range(1, 6):
        tmp = target.with_suffix('.jpg.part')
        try:
            result = subprocess.run(
                ['curl.exe', '--fail', '--silent', '--show-error', '--location', '--max-time', '50', '--output', str(tmp), item['url']],
                capture_output=True, text=True, timeout=55,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f'curl exit {result.returncode}')
            if not tmp.is_file() or tmp.stat().st_size < 1024:
                raise RuntimeError('Downloaded image is missing or too small')
            # A small number of COCO URLs may serve a valid PNG despite the
            # .jpg filename. Normalize such files to JPEG before caching.
            with Image.open(tmp) as image:
                if image.format != 'JPEG':
                    if image.format not in ('PNG', 'WEBP'):
                        raise RuntimeError(f'Unsupported image format: {image.format}')
                    normalized = tmp.with_name(tmp.stem + '.normalized.jpg')
                    image.convert('RGB').save(normalized, format='JPEG', quality=95)
                    normalized.replace(tmp)
            with tmp.open('rb') as f:
                if f.read(2) != b'\xff\xd8':
                    raise RuntimeError('Normalized response is not a JPEG image')
            tmp.replace(target)
            return name, 'downloaded', target.stat().st_size
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            time.sleep(min(attempt, 5))
    return name, f'FAILED: {type(last_error).__name__}: {last_error}', 0

def convert_split(items: list[dict], image_dir: Path, label_dir: Path, split: str) -> dict:
    ids = {int(x['id']) for x in items}
    by_id = {int(x['id']): x for x in items}
    category_by_name = {x['name']: int(x['id']) for x in coco['categories']}
    missing_classes = sorted(set(class_names) - set(category_by_name))
    if missing_classes:
        raise RuntimeError(f'Missing COCO categories: {missing_classes}')
    class_by_category = {category_by_name[name]: index for index, name in enumerate(class_names)}
    rows = {image_id: [] for image_id in ids}
    counts: Counter[int] = Counter()
    crowd = invalid = 0
    for ann in coco['annotations']:
        image_id = int(ann['image_id'])
        if image_id not in ids:
            continue
        if int(ann.get('iscrowd', 0)) == 1:
            crowd += 1
            continue
        category = int(ann['category_id'])
        if category not in class_by_category:
            invalid += 1
            continue
        x, y, w, h = (float(v) for v in ann['bbox'])
        iw, ih = float(by_id[image_id]['width']), float(by_id[image_id]['height'])
        cx, cy, nw, nh = (x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih
        if w <= 0 or h <= 0 or not all(0 <= v <= 1 for v in (cx, cy, nw, nh)):
            invalid += 1
            continue
        cls = class_by_category[category]
        rows[image_id].append(f'{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
        counts[cls] += 1
    for item in items:
        image_id = int(item['id'])
        label_path = label_dir / (Path(item['file_name']).stem + '.txt')
        label_path.write_text('\n'.join(rows[image_id]) + ('\n' if rows[image_id] else ''), encoding='utf-8')
    return {'images': len(items), 'labels': len(list(label_dir.glob('*.txt'))), 'annotations': sum(counts.values()) + crowd, 'yolo_boxes': sum(counts.values()), 'crowd_skipped': crowd, 'invalid_skipped': invalid, 'classes_with_boxes': len(counts)}

for split, items, image_dir in (('train', train_selected, TRAIN_IMAGE_DIR), ('val', val_selected, VAL_IMAGE_DIR)):
    print(f'Downloading {split}: {len(items)} images...', flush=True)
    failures = []
    downloaded = cached = total_bytes = 0
    pending_items = iter(
        item for item in items
        if not (split == 'train' and item['file_name'] in existing_train)
    )
    total = len(items) - (len(existing_train) if split == 'train' else 0)
    index = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        # Bound queued work: never allocate one Future per image up front.
        pending = set()
        for _ in range(min(MAX_IN_FLIGHT, total)):
            try:
                item = next(pending_items)
            except StopIteration:
                break
            pending.add(pool.submit(download_one, item, image_dir))

        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                name, status, size = future.result()
                index += 1
                total_bytes += size
                if status == 'downloaded':
                    downloaded += 1
                elif status == 'cached':
                    cached += 1
                else:
                    failures.append({'file_name': name, 'error': status})
                if index % 250 == 0 or index == total:
                    print(f'{split} progress {index}/{total}; downloaded={downloaded}; cached={cached}; failed={len(failures)}; MB={total_bytes/1024/1024:.1f}', flush=True)
                try:
                    item = next(pending_items)
                except StopIteration:
                    continue
                pending.add(pool.submit(download_one, item, image_dir))
    if failures:
        write_json(ANNOTATION_DIR / f'{split}_download_failures.json', failures)
        raise RuntimeError(f'{len(failures)} {split} image downloads failed; rerun to resume')
    (ANNOTATION_DIR / f'{split}_download_failures.json').unlink(missing_ok=True)
    stats = convert_split(items, image_dir, TRAIN_LABEL_DIR if split == 'train' else VAL_LABEL_DIR, split)
    write_json(ANNOTATION_DIR / f'{split}_summary.json', {**stats, 'downloaded_this_run': downloaded, 'cached': cached, 'downloaded_bytes': total_bytes})
    print(f'{split} labels generated: {stats}', flush=True)

# Final integrity audit: all files present, valid JPEGs/YOLO rows, exact split disjointness.
for split, items, image_dir, label_dir in (
    ('train', train_selected, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR),
    ('val', val_selected, VAL_IMAGE_DIR, VAL_LABEL_DIR),
):
    missing = []
    invalid = []
    for item in items:
        name = item['file_name']
        ip = image_dir / name
        lp = label_dir / (Path(name).stem + '.txt')
        if not ip.is_file() or ip.stat().st_size < 1024 or not lp.is_file():
            missing.append(name)
            continue
        with ip.open('rb') as f:
            if f.read(2) != b'\xff\xd8':
                invalid.append(f'{name}: invalid JPEG')
        for line_no, line in enumerate(lp.read_text(encoding='utf-8').splitlines(), 1):
            fields = line.split()
            try:
                cls = int(fields[0])
                vals = [float(v) for v in fields[1:]]
                if len(fields) != 5 or not 0 <= cls < 80 or not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    raise ValueError('out of range')
            except Exception:
                invalid.append(f'{name}:{line_no}: invalid YOLO label')
    if missing or invalid:
        raise RuntimeError(f'{split} integrity failure: missing={missing[:10]}, invalid={invalid[:10]}')

if train_names & val_names or train_names & test_names or val_names & test_names:
    raise RuntimeError('Final split overlap detected')

names_yaml = '\n'.join(f'  {i}: {name}' for i, name in enumerate(class_names))
yaml = (
    '# Expanded COCO2017 dataset. Test data is held out in a separate dataset and must not be used here.\n'
    'path: F:/Projects/VisionLab/datasets/coco2017_expanded\n'
    'train: images/train2017\n'
    'val: images/val2017\n'
    'names:\n' + names_yaml + '\n'
)
Path('datasets/coco2017_expanded.yaml').write_text(yaml, encoding='utf-8')

summary = {
    'status': 'complete', 'seed': SEED,
    'train_images': len(train_names), 'train_includes_existing_coco128': len(existing_train),
    'validation_images': len(val_names), 'independent_test_images_reserved': len(test_names),
    'previously_used_official_val_images_excluded': len(official_val_names),
    'train_val_overlap': 0, 'train_test_overlap': 0, 'val_test_overlap': 0,
    'train_classes': 80, 'validation_classes': 80,
    'dataset_yaml': 'datasets/coco2017_expanded.yaml',
    'selection_manifest': str(selection_path),
}
write_json(ANNOTATION_DIR / 'preparation_summary.json', summary)
print(json.dumps(summary, indent=2), flush=True)
