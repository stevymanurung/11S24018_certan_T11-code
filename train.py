"""Melatih model YOLO untuk kelas melon, daun, dan buah rusak."""
from pathlib import Path
import os
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # Pastikan path relatif pada data.yaml selalu mengarah ke folder proyek.
DATA = ROOT / "dataset" / "data.yaml"
WEIGHTS = ROOT / "yolo26n.pt"

if not DATA.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA}")
if not WEIGHTS.exists():
    raise FileNotFoundError(f"Bobot awal tidak ditemukan: {WEIGHTS}")

model = YOLO(str(WEIGHTS))
model.train(
    data=str(DATA),
    epochs=50,
    imgsz=640,
    batch=4,
    project=str(ROOT / "runs" / "melon"),
    name="baseline",
    exist_ok=True,
    workers=0,
    patience=15,
)
print("Training selesai. Bobot terbaik disimpan di:")
print(ROOT / "runs" / "melon" / "baseline" / "weights" / "best.pt")
