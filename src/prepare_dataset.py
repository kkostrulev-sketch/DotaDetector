"""Convert DOTA TXT annotations to Ultralytics YOLO OBB format."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

DOTA_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]

CLASS_TO_ID = {name: index for index, name in enumerate(DOTA_CLASSES)}


def parse_dota_line(line: str) -> tuple[list[float], str] | None:
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    try:
        coords = [float(parts[index]) for index in range(8)]
    except ValueError:
        return None
    class_name = parts[8]
    if class_name not in CLASS_TO_ID:
        return None
    return coords, class_name


def convert_label_file(
    source_txt: Path,
    target_txt: Path,
    image_width: int,
    image_height: int,
) -> int:
    lines: list[str] = []
    for raw_line in source_txt.read_text(encoding="utf-8").splitlines():
        parsed = parse_dota_line(raw_line)
        if parsed is None:
            continue
        coords, class_name = parsed
        normalized = []
        for index, value in enumerate(coords):
            if index % 2 == 0:
                normalized.append(value / image_width)
            else:
                normalized.append(value / image_height)
        class_id = CLASS_TO_ID[class_name]
        coord_text = " ".join(f"{value:.6f}" for value in normalized)
        lines.append(f"{class_id} {coord_text}")

    target_txt.parent.mkdir(parents=True, exist_ok=True)
    target_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def get_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def prepare_split(
    images_dir: Path,
    labels_dir: Path,
    output_images: Path,
    output_labels: Path,
) -> dict[str, int]:
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    converted_images = 0
    converted_objects = 0

    for image_path in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        width, height = get_image_size(image_path)
        target_image = output_images / image_path.name
        target_label = output_labels / f"{image_path.stem}.txt"

        if not target_image.exists():
            shutil.copy2(image_path, target_image)

        converted_objects += convert_label_file(label_path, target_label, width, height)
        converted_images += 1

    return {"images": converted_images, "objects": converted_objects}


def write_dataset_yaml(output_root: Path, dataset_yaml: Path) -> None:
    payload = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(DOTA_CLASSES)},
    }
    dataset_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DOTAv1 dataset for YOLO OBB training")
    parser.add_argument(
        "--source-images",
        type=Path,
        required=True,
        help="Root folder with train/val/test image subfolders",
    )
    parser.add_argument(
        "--source-labels",
        type=Path,
        required=True,
        help="Root folder with train/val/test DOTA TXT label subfolders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DOTAv1"),
        help="Output dataset root",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("DOTAv1.yaml"),
        help="Path to generated Ultralytics dataset yaml",
    )
    args = parser.parse_args()

    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        images_dir = args.source_images / split
        labels_dir = args.source_labels / split
        if not images_dir.exists() or not labels_dir.exists():
            continue
        summary[split] = prepare_split(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_images=args.output / "images" / split,
            output_labels=args.output / "labels" / split,
        )

    write_dataset_yaml(args.output, args.dataset_yaml)
    print(f"Dataset prepared at: {args.output.resolve()}")
    print(f"Dataset yaml: {args.dataset_yaml.resolve()}")
    for split, stats in summary.items():
        print(f"{split}: {stats['images']} images, {stats['objects']} objects")


if __name__ == "__main__":
    main()