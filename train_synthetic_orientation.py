"""Fine-tune a small 0/180 text orientation classifier and report 1-Brier.

Run from the Avito project with: python train_synthetic_orientation.py
All paths and hyperparameters are constants below. The validation split is
synthetic and measures pipeline performance, not the hidden Avito score.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "old_train_data" / "synthetic_orientation"
OUTPUT_DIR = PROJECT_DIR / "models" / "synthetic_mobilenetv3_small"

SEED = 20260925
HEIGHT, WIDTH = 96, 320
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class OrientationDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool) -> None:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            self.rows = list(csv.DictReader(file))
        self.augment = augment
        self.color = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.10)
        self.to_tensor = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(DATA_DIR / row["image_path"]) as source:
            image = source.convert("RGB")
        image = ImageOps.contain(image, (WIDTH, HEIGHT), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (127, 127, 127))
        canvas.paste(image, ((WIDTH - image.width) // 2,
                             (HEIGHT - image.height) // 2))
        if self.augment:
            canvas = self.color(canvas)
        return self.to_tensor(canvas), torch.tensor(
            float(row["y_180"]), dtype=torch.float32)


def make_loader(csv_path: Path, augment: bool, seed: int) -> DataLoader:
    return DataLoader(
        OrientationDataset(csv_path, augment), batch_size=BATCH_SIZE,
        shuffle=augment, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def make_model() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict[str, float]:
    model.eval()
    sum_brier = 0.0
    correct = 0
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images).flatten()
        probs = logits.sigmoid()
        sum_brier += torch.sum((probs - labels).square()).item()
        correct += torch.sum((probs >= 0.5) == (labels >= 0.5)).item()
        n += labels.numel()
    brier = sum_brier / n
    return {"n": n, "brier": brier, "score_1_minus_brier": 1.0 - brier,
            "accuracy": correct / n}


def main() -> None:
    if not (DATA_DIR / "train.csv").is_file() or not (DATA_DIR / "val.csv").is_file():
        raise FileNotFoundError(f"Expected train.csv and val.csv in {DATA_DIR}")
    seed_everything(SEED)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; data: {DATA_DIR}", flush=True)

    train_loader = make_loader(DATA_DIR / "train.csv", True, SEED)
    val_loader = make_loader(DATA_DIR / "val.csv", False, SEED + 1)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE / 20)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_brier = float("inf")
    stale_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        started = time.perf_counter()
        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images).flatten()
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.numel()
            total_seen += labels.numel()
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss={total_loss / total_seen:.5f}", flush=True)

        metrics = validate(model, val_loader, device)
        metrics.update({"epoch": epoch, "train_bce": total_loss / total_seen,
                        "seconds": time.perf_counter() - started})
        history.append(metrics)
        print("validation", json.dumps(metrics, ensure_ascii=False), flush=True)
        if metrics["brier"] < best_brier:
            best_brier = metrics["brier"]
            stale_epochs = 0
            torch.save({"architecture": "mobilenet_v3_small",
                        "input_size": [HEIGHT, WIDTH], "seed": SEED,
                        "epoch": epoch, "model_state": model.state_dict()},
                       OUTPUT_DIR / "best.pt")
        else:
            stale_epochs += 1
        (OUTPUT_DIR / "metrics.json").write_text(
            json.dumps({"data": str(DATA_DIR), "best_brier": best_brier,
                        "history": history}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        scheduler.step()
        if stale_epochs >= EARLY_STOP_PATIENCE:
            print("Early stopping", flush=True)
            break

    print(f"Best 1-Brier: {1.0 - best_brier:.8f}", flush=True)
    print(f"Weights: {OUTPUT_DIR / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
