"""Train the rain drop YOLOv11 model locally from a labeled dataset in Azure Blob Storage.

Reads TRAINING_STORAGE_CONNECTION_STRING from the environment or a .env file.

Expected dataset layout under the blob prefix (standard Ultralytics format):
    data.yaml
    images/{train,val}/...
    labels/{train,val}/...

Example:
    uv run python scripts/train_rain_model.py --container <container>
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import yaml
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
from ultralytics import YOLO

logger = logging.getLogger("train_rain_model")


def download_dataset(
    connection_string: str, container: str, prefix: str, dest: Path
) -> None:
    container_client = ContainerClient.from_connection_string(
        connection_string, container_name=container
    )
    count = 0
    for blob in container_client.list_blobs(name_starts_with=prefix):
        relative = Path(blob.name[len(prefix) :].lstrip("/"))
        if not blob.name or blob.name.endswith("/") or not relative.parts:
            continue
        target = (dest / relative).resolve()
        if not target.is_relative_to(dest.resolve()):
            raise ValueError(f"Blob path escapes destination: {blob.name}")
        if target.exists() and target.stat().st_size == blob.size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            container_client.download_blob(blob.name).readinto(f)
        count += 1
        if count % 100 == 0:
            logger.info(f"Downloaded {count} files")
    logger.info(f"Downloaded {count} new files to {dest}")


def prepare_data_yaml(dataset_dir: Path) -> Path:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. The dataset must contain a data.yaml."
        )
    config = yaml.safe_load(data_yaml.read_text())
    config["path"] = str(dataset_dir.resolve())  # make paths local
    local_yaml = dataset_dir / "data.local.yaml"
    local_yaml.write_text(yaml.safe_dump(config))
    return local_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--container", default="rain-drop-training-dataset")
    parser.add_argument("--prefix", default="", help="Dataset folder in the container")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/rain_dataset"))
    parser.add_argument("--output", type=Path, default=Path("rain_detection_model.pt"))
    parser.add_argument(
        "--base-model", default="yolo11m.pt", help="e.g. yolo11s.pt or yolo11m.pt"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="", help="'' = auto (cuda/mps/cpu), or e.g. 'cpu', 'mps'"
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_dotenv()

    if not args.skip_download:
        connection_string = os.environ.get("TRAINING_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise SystemExit("TRAINING_STORAGE_CONNECTION_STRING is not set")
        download_dataset(
            connection_string, args.container, args.prefix, args.dataset_dir
        )
    data_yaml = prepare_data_yaml(args.dataset_dir)

    model = YOLO(args.base_model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        device=args.device or None,
        project=str(Path("runs/rain").resolve()),  # relative paths get a task prefix
        name="train",
        exist_ok=True,
    )

    assert model.trainer is not None
    best = Path(model.trainer.best)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.output)
    logger.info(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()
