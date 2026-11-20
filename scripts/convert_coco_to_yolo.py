"""Convert COCO detection annotations to Ultralytics YOLO labels.

The converter preserves the original validation images and writes one label
file per image. COCO category IDs are mapped by class name, not by assuming
that category IDs are contiguous.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def convert(coco_root: Path, classes_file: Path, overwrite: bool = False) -> dict[str, Any]:
    """Convert COCO validation annotations into normalized YOLO label files.

    Category IDs are translated through class names, crowd annotations are
    excluded, and existing non-empty labels are protected unless overwrite is
    explicitly requested.
    """
    annotation_file = coco_root / "annotations" / "instances_val2017.json"
    image_dir = coco_root / "images" / "val2017"
    label_dir = coco_root / "labels" / "val2017"
    if not annotation_file.is_file() or not image_dir.is_dir():
        raise FileNotFoundError("Expected annotations/instances_val2017.json and images/val2017")

    class_names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(class_names) != 80:
        raise ValueError(f"Expected exactly 80 class names, found {len(class_names)}")

    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    # COCO category IDs are not guaranteed to be contiguous or ordered like the
    # model's class indices, so map through the canonical classes file by name.
    category_by_name = {item["name"]: int(item["id"]) for item in data["categories"]}
    missing = sorted(set(class_names) - set(category_by_name))
    if missing:
        raise ValueError(f"COCO categories missing from classes file: {missing}")
    class_id_by_category = {category_by_name[name]: index for index, name in enumerate(class_names)}

    if label_dir.exists() and any(label_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Labels already exist; pass --overwrite to replace them: {label_dir}")
    label_dir.mkdir(parents=True, exist_ok=True)

    image_by_id = {int(item["id"]): item for item in data["images"]}
    rows: dict[int, list[str]] = {image_id: [] for image_id in image_by_id}
    skipped_crowd = 0
    skipped_invalid = 0
    class_counts: Counter[int] = Counter()

    # Convert COCO [x, y, width, height] pixel boxes into YOLO normalized
    # [center_x, center_y, width, height] coordinates for each image.
    for annotation in data["annotations"]:
        if int(annotation.get("iscrowd", 0)) == 1:
            skipped_crowd += 1
            continue
        image_id = int(annotation["image_id"])
        if image_id not in image_by_id or int(annotation["category_id"]) not in class_id_by_category:
            skipped_invalid += 1
            continue

        image = image_by_id[image_id]
        x, y, width, height = (float(value) for value in annotation["bbox"])
        image_width, image_height = float(image["width"]), float(image["height"])
        if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
            skipped_invalid += 1
            continue

        # Convert COCO's top-left xywh box to normalized YOLO center xywh.
        center_x = (x + width / 2) / image_width
        center_y = (y + height / 2) / image_height
        norm_width, norm_height = width / image_width, height / image_height
        values = (center_x, center_y, norm_width, norm_height)
        if not all(0 <= value <= 1 for value in values):
            skipped_invalid += 1
            continue

        class_id = class_id_by_category[int(annotation["category_id"])]
        rows[image_id].append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values))
        class_counts[class_id] += 1

    missing_images: list[str] = []
    for image_id, image in image_by_id.items():
        filename = str(image["file_name"])
        if not (image_dir / filename).is_file():
            missing_images.append(filename)
            continue
        (label_dir / f"{Path(filename).stem}.txt").write_text(
            "\n".join(rows[image_id]) + ("\n" if rows[image_id] else ""),
            encoding="utf-8",
        )

    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} image files; first: {missing_images[:5]}")

    return {
        "images": len(image_by_id),
        "label_files": len(list(label_dir.glob("*.txt"))),
        "annotations_written": sum(class_counts.values()),
        "skipped_crowd": skipped_crowd,
        "skipped_invalid": skipped_invalid,
        "classes_with_annotations": len(class_counts),
        "class_counts": {class_names[index]: count for index, count in sorted(class_counts.items())},
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
    }


def main() -> None:
    """Parse conversion options and report the COCO-to-YOLO dataset conversion summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=Path("datasets/coco2017"))
    parser.add_argument("--classes", type=Path, default=Path("datasets/classes.txt"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(convert(args.coco_root, args.classes, args.overwrite), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
