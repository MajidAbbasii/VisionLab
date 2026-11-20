"""Compare Python and C# detections on the same images using class-aware IoU matching.

The script writes Python reference detections and a JSON parity report. It does
not retrain or modify either runtime's model or inference implementation.
"""
from __future__ import annotations
import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/predictions/runtime-batch-smoke-input'
CSHARP = ROOT / 'results/predictions/runtime-batch-smoke-output/json'
OUT = ROOT / 'results/predictions/python-parity-reference'
REPORT = ROOT / 'results/reports/csharp_python_parity_2026-11-20.json'
CONF, IOU = 0.25, 0.45


def iou_xyxy(a, b):
    """Return intersection-over-union for two boxes in [x1, y1, x2, y2] form."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    """Run both inference paths on the prepared image batch and save a parity report."""
    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in INPUT.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'})
    if not images:
        raise SystemExit(f'No images found: {INPUT}')
    model = YOLO(str(ROOT / 'models/exported/best.onnx'), task='detect')
    results = model.predict(source=[str(p) for p in images], imgsz=640, conf=CONF, iou=IOU,
                            device='cpu', save=False, verbose=False)
    # Match each Python detection to the best compatible C# detection using
    # class ID and IoU; this measures runtime parity rather than model accuracy.
    rows = []
    for image, result in zip(images, results):
        with (CSHARP / f'{image.stem}.json').open(encoding='utf-8') as f:
            csharp = json.load(f)['detections']
        python = []
        if result.boxes is not None:
            for box in result.boxes:
                xyxy = [float(x) for x in box.xyxy[0].cpu().tolist()]
                cls = int(box.cls.item())
                python.append({'class_id': cls, 'confidence': float(box.conf.item()), 'xyxy': xyxy})
        py_path = OUT / f'{image.stem}.json'
        py_path.write_text(json.dumps({'source': str(image), 'confidence_threshold': CONF,
            'iou_threshold': IOU, 'detections': python}, ensure_ascii=False, indent=2), encoding='utf-8')
        c_boxes = [{'class_id': int(d['ClassId']), 'confidence': float(d['Confidence']),
                    'xyxy': [float(d['X']), float(d['Y']), float(d['X']+d['Width']), float(d['Y']+d['Height'])]}
                   for d in csharp]
        candidates = []
        for ci, c in enumerate(c_boxes):
            for pi, p in enumerate(python):
                if c['class_id'] == p['class_id']:
                    overlap = iou_xyxy(c['xyxy'], p['xyxy'])
                    if overlap >= 0.5:
                        candidates.append((overlap, ci, pi))
        candidates.sort(reverse=True)
        used_c, used_p, matches = set(), set(), []
        for overlap, ci, pi in candidates:
            if ci not in used_c and pi not in used_p:
                used_c.add(ci); used_p.add(pi)
                matches.append({'iou': overlap, 'confidence_abs_diff': abs(c_boxes[ci]['confidence']-python[pi]['confidence']),
                                'class_id': c_boxes[ci]['class_id']})
        rows.append({'image': image.name, 'csharp_count': len(c_boxes), 'python_count': len(python),
                     'matched_count': len(matches), 'unmatched_csharp': len(c_boxes)-len(used_c),
                     'unmatched_python': len(python)-len(used_p),
                     'mean_matched_iou': sum(m['iou'] for m in matches)/len(matches) if matches else None,
                     'max_confidence_abs_diff': max((m['confidence_abs_diff'] for m in matches), default=None)})
    total_c = sum(x['csharp_count'] for x in rows)
    total_p = sum(x['python_count'] for x in rows)
    total_m = sum(x['matched_count'] for x in rows)
    ious = [x['mean_matched_iou'] for x in rows if x['mean_matched_iou'] is not None]
    report = {'model': 'models/exported/best.onnx', 'provider': 'CPU', 'images': len(rows),
              'confidence_threshold': CONF, 'iou_threshold': IOU, 'matching_rule': 'same class, greedy one-to-one, IoU >= 0.50',
              'csharp_total_detections': total_c, 'python_total_detections': total_p,
              'matched_detections': total_m, 'unmatched_csharp': total_c-total_m,
              'unmatched_python': total_p-total_m,
              'matched_fraction_of_union': total_m/(total_c+total_p-total_m) if total_c+total_p-total_m else 1.0,
              'mean_per_image_matched_iou': sum(ious)/len(ious) if ious else None,
              'per_image': rows}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'REPORT={REPORT.relative_to(ROOT)}')

if __name__ == '__main__':
    main()
