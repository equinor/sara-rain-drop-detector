"""Split a Label Studio YOLO export into train/val/test and write data.yaml."""

import argparse
import random
import re
import shutil
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TIMESTAMP_PATTERN = re.compile(r"__Image__(\d+)")


def group_by_time(images: list[Path], max_gap: int) -> list[list[Path]]:
    """Group images whose capture times are within max_gap seconds of each other."""
    timed: list[tuple[int, Path]] = []
    groups: list[list[Path]] = []
    for image in images:
        match = TIMESTAMP_PATTERN.search(image.name)
        if match:
            timed.append((int(match.group(1)), image))
        else:
            groups.append([image])
    previous = None
    for timestamp, image in sorted(timed):
        if previous is None or timestamp - previous > max_gap:
            groups.append([])
        groups[-1].append(image)
        previous = timestamp
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/rain_dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--group-gap",
        type=int,
        default=300,
        help="Images less than this many seconds apart are kept in the same split",
    )
    args = parser.parse_args()

    source = args.source.expanduser()
    images = sorted(
        p for p in (source / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"No images found in {source / 'images'}")

    groups = group_by_time(images, args.group_gap)
    print(f"{len(images)} images in {len(groups)} time groups")
    random.Random(args.seed).shuffle(groups)

    # Whole groups go to one split, so sizes only approximate the fractions.
    n_test = round(len(images) * args.test_fraction)
    n_val = round(len(images) * args.val_fraction)
    splits: dict[str, list[Path]] = {"test": [], "val": [], "train": []}
    for group in groups:
        if len(splits["test"]) < n_test:
            splits["test"].extend(group)
        elif len(splits["val"]) < n_val:
            splits["val"].extend(group)
        else:
            splits["train"].extend(group)

    if args.output.exists():
        shutil.rmtree(args.output)

    for split, files in splits.items():
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)
        for image in files:
            shutil.copy(image, args.output / "images" / split / image.name)
            label = source / "labels" / f"{image.stem}.txt"
            target = args.output / "labels" / split / label.name
            if label.exists():
                shutil.copy(label, target)
            else:
                target.touch()  # empty label = background image
        print(f"{split}: {len(files)}")

    class_names = [
        line.strip()
        for line in (source / "classes.txt").read_text().splitlines()
        if line.strip()
    ]
    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(class_names)),
    }
    (args.output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
