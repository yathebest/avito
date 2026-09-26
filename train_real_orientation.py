"""Fine-tune the compact orientation model on real HierText crop pairs.

Train and validation come from the official, disjoint HierText splits. The
base-upright labels are inferred from ordered word geometry, not human-verified
orientation labels; the 180-degree counterpart is an exact pixel rotation.

Run from the Avito project directory: python train_real_orientation.py
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
from torchvision.models import mobilenet_v3_small


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_DIR / "old_train_data" / "real_orientation_train.csv"
VAL_CSV = PROJECT_DIR / "old_validation_data" / "orientation_val" / "val.csv"
WARMSTART = PROJECT_DIR / "models" / "synthetic_mobilenetv3_small" / "best.pt"
OUTPUT_DIR = PROJECT_DIR / "models" / "real_hiertext_mobilenetv3_small"

SEED = 20260926
HEIGHT, WIDTH = 96, 320
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
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


class RealPairsDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool) -> None:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            self.rows = list(csv.DictReader(file))
        self.augment = augment
        self.color = transforms.ColorJitter(
            brightness=0.20, contrast=0.20, saturation=0.12)
        self.to_tensor = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        source_path = PROJECT_DIR / row["image_relpath"]
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        if int(row["rotation_deg"]) == 180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        image = ImageOps.contain(image, (WIDTH, HEIGHT), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (127, 127, 127))
        canvas.paste(image, ((WIDTH - image.width) // 2,
                             (HEIGHT - image.height) // 2))
        if self.augment:
            canvas = self.color(canvas)
        return self.to_tensor(canvas), torch.tensor(
            float(row["y_180"]), dtype=torch.float32)


def make_loader(path: Path, augment: bool, seed: int) -> DataLoader:
    return DataLoader(RealPairsDataset(path, augment), batch_size=BATCH_SIZE,
                      shuffle=augment, num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=NUM_WORKERS > 0,
                      generator=torch.Generator().manual_seed(seed))


def make_model() -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    checkpoint = torch.load(WARMSTART, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return model


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict[str, float]:
    model.eval()
    brier_sum = 0.0
    correct = 0
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        probs = model(images).flatten().sigmoid()
        brier_sum += torch.sum((probs - labels).square()).item()
        correct += torch.sum((probs >= 0.5) == (labels >= 0.5)).item()
        n += labels.numel()
    return {"n": n, "brier": brier_sum / n,
            "score_1_minus_brier": 1.0 - brier_sum / n,
            "accuracy": correct / n}


def main() -> None:
    for path in (TRAIN_CSV, VAL_CSV, WARMSTART):
        if not path.is_file():
            raise FileNotFoundError(path)
    seed_everything(SEED)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(TRAIN_CSV, True, SEED)
    val_loader = make_loader(VAL_CSV, False, SEED + 1)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE / 20)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}; train={len(train_loader.dataset)}; "
          f"validation={len(val_loader.dataset)}", flush=True)
    initial = validate(model, val_loader, device)
    print("warmstart validation", json.dumps(initial), flush=True)
    history = []
    best_brier = float("inf")
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        started = time.perf_counter()
        sum_loss = 0.0
        seen = 0
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
            sum_loss += loss.item() * labels.numel()
            seen += labels.numel()
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} "
                      f"bce={sum_loss / seen:.5f}", flush=True)
        metrics = validate(model, val_loader, device)
        metrics.update({"epoch": epoch, "train_bce": sum_loss / seen,
                        "seconds": time.perf_counter() - started})
        history.append(metrics)
        print("validation", json.dumps(metrics), flush=True)
        if metrics["brier"] < best_brier:
            best_brier = metrics["brier"]
            stale = 0
            torch.save({"architecture": "mobilenet_v3_small",
                        "input_size": [HEIGHT, WIDTH], "seed": SEED,
                        "epoch": epoch, "model_state": model.state_dict()},
                       OUTPUT_DIR / "best.pt")
        else:
            stale += 1
        (OUTPUT_DIR / "metrics.json").write_text(json.dumps(
            {"train_csv": str(TRAIN_CSV), "val_csv": str(VAL_CSV),
             "warmstart_validation": initial, "best_brier": best_brier,
             "history": history}, indent=2), encoding="utf-8")
        scheduler.step()
        if stale >= EARLY_STOP_PATIENCE:
            print("Early stopping", flush=True)
            break
    print(f"Best 1-Brier: {1.0 - best_brier:.8f}", flush=True)
    print(f"Weights: {OUTPUT_DIR / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
