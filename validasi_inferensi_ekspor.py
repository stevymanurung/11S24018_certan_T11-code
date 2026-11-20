"""Validasi, inferensi, benchmark sederhana, dan ekspor model YOLO."""
from pathlib import Path
import os
import argparse
import time
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # Pastikan dataset dan sample ditemukan dari lokasi proyek.
DATA = ROOT / "dataset" / "data.yaml"
CUSTOM = ROOT / "runs" / "melon" / "baseline" / "weights" / "best.pt"
PRETRAINED = ROOT / "yolo26n.pt"


def load_model(weights: str | None, allow_pretrained_fallback: bool = True) -> YOLO:
    path = Path(weights).expanduser().resolve() if weights else CUSTOM
    if not path.exists():
        if weights is None and allow_pretrained_fallback and PRETRAINED.exists():
            print("Bobot custom belum ada; memakai yolo26n.pt untuk demo inferensi.")
            path = PRETRAINED
        else:
            raise FileNotFoundError(
                f"Bobot model tidak ditemukan: {path}\nJalankan `python train.py` terlebih dahulu."
            )
    print(f"Memuat model: {path}")
    return YOLO(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["val", "image", "video", "webcam", "export", "benchmark"])
    parser.add_argument("--source", help="Path gambar/video; wajib untuk action image/video")
    parser.add_argument("--weights", help="Path bobot .pt; default best.pt jika tersedia")
    parser.add_argument("--conf", type=float, default=0.25, help="Ambang confidence (0-1)")
    args = parser.parse_args()

    # Validasi dan ekspor untuk tugas ini harus memakai bobot hasil training custom.
    model = load_model(args.weights, allow_pretrained_fallback=args.action not in ("val", "export"))
    if args.action == "val":
        if not DATA.exists():
            raise FileNotFoundError(DATA)
        metrics = model.val(data=str(DATA), imgsz=640, plots=True)
        print("\nMetrik validasi:")
        print(f"mAP50      : {metrics.box.map50:.4f}")
        print(f"mAP50-95   : {metrics.box.map:.4f}")
        print(f"Precision  : {metrics.box.mp:.4f}")
        print(f"Recall     : {metrics.box.mr:.4f}")
    elif args.action in ("image", "video"):
        if not args.source:
            parser.error("--source wajib diisi untuk image/video")
        src = Path(args.source).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"Sumber tidak ditemukan: {src}")
        results = model.predict(source=str(src), conf=args.conf, save=True)
        for result in results:
            print(f"Hasil: {result.save_dir}")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
                print(f"{model.names[cls_id]} | confidence={score:.3f} | bbox={xyxy}")
    elif args.action == "webcam":
        model.predict(source=0, conf=args.conf, show=True)
    elif args.action == "export":
        exported = model.export(format="onnx", imgsz=640)
        print(f"Model ONNX berhasil diekspor: {exported}")
    elif args.action == "benchmark":
        sample = ROOT / "sample.jpg"
        if not sample.exists():
            raise FileNotFoundError(sample)
        start = time.perf_counter()
        model.predict(source=str(sample), conf=args.conf, verbose=False)
        elapsed = time.perf_counter() - start
        print(f"Waktu satu inferensi (termasuk overhead pemanggilan): {elapsed * 1000:.2f} ms")
        print(f"Perkiraan FPS satu gambar: {1 / elapsed:.2f}")
        print("Catatan: ukur berulang pada perangkat target untuk benchmark yang representatif.")


if __name__ == "__main__":
    main()
