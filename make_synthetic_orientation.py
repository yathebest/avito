"""Create exactly labelled 0/180 text-line crops for orientation training.

Run with: python make_synthetic_orientation.py

The text is rendered upright first. Its 180-degree counterpart is made from
the same pixels, so neither label depends on OCR or geometry guesses. These
synthetic images are for training; evaluate on independent real crops.
"""

from __future__ import annotations

import csv
import io
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "old_train_data" / "synthetic_orientation"
WINDOWS_DIR = Path(os.environ.get("WINDIR", "C:/Windows"))
FONT_DIR = Path(os.environ.get("AVITO_FONT_DIR", WINDOWS_DIR / "Fonts"))
FONT_NAMES = (
    "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf",
    "times.ttf", "timesbd.ttf", "tahoma.ttf", "tahomabd.ttf",
    "verdana.ttf", "verdanab.ttf", "segoeui.ttf", "seguisb.ttf",
    "georgia.ttf", "georgiab.ttf", "cour.ttf", "courbd.ttf",
)

TRAIN_BASE_IMAGES = 30_000
VAL_BASE_IMAGES = 2_000  # Synthetic sanity check; real validation is essential.
SEED = 20260925
MAX_TILT_DEG = 7.0  # Both 0 and 180 classes may have a little scene-text tilt.

RU_ADJECTIVES = (
    "новый", "новая", "большой", "удобный", "красивый", "детский",
    "домашний", "зимний", "летний", "винтажный", "компактный",
    "кожаный", "деревянный", "оригинальный", "фирменный", "яркий",
    "мягкий", "теплый", "профессиональный", "складной", "легкий",
    "белый", "черный", "красный", "синий", "зеленый", "серый",
)
RU_NOUNS = (
    "телефон", "ноутбук", "планшет", "монитор", "принтер", "камера",
    "объектив", "велосипед", "самокат", "рюкзак", "костюм", "куртка",
    "платье", "свитер", "рубашка", "диван", "стол", "стул", "шкаф",
    "кровать", "матрас", "лампа", "часы", "колонка", "наушники",
    "книга", "игрушка", "картина", "зеркало", "ваза", "кофеварка",
    "холодильник", "пылесос", "кроссовки", "ботинки", "пальто",
    "сумка", "чехол", "зарядка", "клавиатура", "мышь", "микрофон",
    "цветы", "подарок", "аксессуары", "запчасти", "инструменты",
)
RU_SHORT = (
    "Продам", "Скидка", "Доставка", "Гарантия", "Размер", "Цена",
    "Новинка", "Оригинал", "В наличии", "Без торга", "Хорошее состояние",
    "Сделано в России", "Москва", "Санкт-Петербург", "Казань",
    "Распродажа", "Акция", "Комплект", "Бесплатно", "Для дома",
)
LATIN = (
    "Samsung", "iPhone", "Apple", "Nike", "Adidas", "Bosch", "Sony",
    "Canon", "Lenovo", "ASUS", "Intel", "Nokia", "Mango", "Coffee",
    "SALE", "NEW", "Original", "Made in Italy", "Best price", "ONLINE",
    "Wireless speaker", "Fresh market", "Limited edition", "No. 1",
)


def make_text(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.60:
        words = [rng.choice(RU_ADJECTIVES), rng.choice(RU_NOUNS)]
        if rng.random() < 0.35:
            words.append(str(rng.randint(2, 999)))
        return " ".join(words)
    if kind < 0.78:
        return rng.choice(RU_SHORT)
    if kind < 0.92:
        return rng.choice(LATIN)
    return f"{rng.choice(RU_NOUNS).capitalize()} {rng.randint(100, 99999)} руб."


def choose_font(rng: random.Random, paths: list[Path]) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(rng.choice(paths)), rng.randint(14, 54))


def make_upright_crop(text: str, rng: random.Random,
                      font_paths: list[Path]) -> tuple[Image.Image, float]:
    font = choose_font(rng, font_paths)
    bbox = font.getbbox(text)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = rng.randint(3, 16), rng.randint(2, 10)
    if rng.random() < 0.42:
        bg = tuple(rng.randint(190, 255) for _ in range(3))
        fg = tuple(rng.randint(0, 70) for _ in range(3))
    elif rng.random() < 0.73:
        bg = tuple(rng.randint(0, 65) for _ in range(3))
        fg = tuple(rng.randint(190, 255) for _ in range(3))
    else:
        bg = tuple(rng.randint(45, 210) for _ in range(3))
        fg = (255, 255, 255) if sum(bg) < 380 else (10, 10, 10)

    img = Image.new("RGB", (max(12, text_w + 2 * pad_x),
                            max(8, text_h + 2 * pad_y)), bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text,
              font=font, fill=fg, stroke_width=0)

    tilt = rng.uniform(-MAX_TILT_DEG, MAX_TILT_DEG)
    img = img.rotate(tilt, resample=Image.Resampling.BICUBIC,
                     expand=True, fillcolor=bg)
    if rng.random() < 0.35:
        scale = rng.uniform(0.55, 0.85)
        img = img.resize((max(12, round(img.width * scale)),
                          max(8, round(img.height * scale))),
                         Image.Resampling.LANCZOS)
    if rng.random() < 0.18:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.55)))
    return img, tilt


def save_pair(base: Image.Image, prefix: str, folder: Path,
              rng: random.Random, writer: csv.writer, text: str,
              tilt: float, split: str) -> None:
    quality = rng.randint(65, 95)
    # Encode once so 180 is applied to the very same degraded source pixels.
    buffer = io.BytesIO()
    base.save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        upright = encoded.convert("RGB")
    for label, image in ((0, upright),
                         (1, upright.transpose(Image.Transpose.ROTATE_180))):
        name = f"{prefix}_{label}.png"
        image.save(folder / name)
        writer.writerow((f"images/{split}/{name}", label, prefix, text,
                         f"{tilt:.3f}"))


def main() -> None:
    fonts = [FONT_DIR / name for name in FONT_NAMES
             if (FONT_DIR / name).is_file()]
    if not fonts:
        raise FileNotFoundError(f"No expected Cyrillic-capable fonts in {FONT_DIR}")
    print(f"Using {len(fonts)} fonts", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for split, amount in (("train", TRAIN_BASE_IMAGES),
                          ("val", VAL_BASE_IMAGES)):
        folder = OUTPUT_DIR / "images" / split
        folder.mkdir(parents=True, exist_ok=True)
        with (OUTPUT_DIR / f"{split}.csv").open("w", encoding="utf-8",
                                                newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("image_path", "y_180", "source_id", "text",
                             "base_tilt_deg"))
            for index in range(amount):
                text = make_text(rng)
                upright, tilt = make_upright_crop(text, rng, fonts)
                save_pair(upright, f"{split}_{index:06d}", folder,
                          rng, writer, text, tilt, split)
                if (index + 1) % 5000 == 0:
                    print(f"{split}: {index + 1}/{amount} base crops", flush=True)
        print(f"{split}: {amount * 2} labelled images", flush=True)


if __name__ == "__main__":
    main()
