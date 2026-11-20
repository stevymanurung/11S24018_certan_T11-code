"""Demo slide: validasi, inferensi gambar/video/webcam, ekspor ONNX, benchmark YOLO.

Default memakai yolo26n.pt (model pretrained COCO) yang sudah disertakan.
Untuk model lain, gunakan --weights path/model.pt.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import psutil
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
DEFAULT_WEIGHTS = ROOT / "yolo26n.pt"
SAMPLE_IMAGE = ROOT / "sample.jpg"
COCO_CAT_ID = 15  # ID kelas "cat" pada label COCO yang digunakan model pretrained.


def load_model(weights: str | None) -> YOLO:
    path = Path(weights).expanduser().resolve() if weights else DEFAULT_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(
            f"File bobot tidak ditemukan: {path}\n"
            "Letakkan yolo26n.pt di folder proyek atau berikan --weights."
        )
    print(f"Memuat model: {path}")
    model = YOLO(str(path))
    print("Kelas model:", model.names)
    if isinstance(model.names, dict) and COCO_CAT_ID in model.names:
        print(f"Kelas kucing tersedia: ID {COCO_CAT_ID} = {model.names[COCO_CAT_ID]}")
    return model


def prediction_classes(cat_only: bool) -> list[int] | None:
    return [COCO_CAT_ID] if cat_only else None


def do_validation(model: YOLO, data: str) -> None:
    """Validasi memakai dataset COCO8 bawaan Ultralytics (diunduh otomatis bila perlu)."""
    print(f"\nValidasi model dengan dataset: {data}")
    metrics = model.val(data=data, imgsz=640, plots=True)
    print("\nHASIL VALIDASI")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print("Folder hasil validasi: periksa folder runs/detect/val*")


def do_predict(model: YOLO, source: str, cat_only: bool, show: bool = False) -> None:
    src = Path(source).expanduser()
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        raise FileNotFoundError(
            f"Sumber tidak ditemukan: {src}\n"
            "Untuk video, letakkan video.mp4 di folder utama proyek atau ubah --source."
        )
    results = model.predict(
        source=str(src),
        save=True,
        show=show,
        conf=0.25,
        classes=prediction_classes(cat_only),
    )
    for result in results:
        print(f"Hasil tersimpan di: {result.save_dir}")
        if result.boxes is None or len(result.boxes) == 0:
            print("Tidak ada objek yang terdeteksi pada frame/gambar ini.")
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coords = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            print(f"{model.names[class_id]} | confidence={confidence:.3f} | bbox={coords}")


def do_webcam(model: YOLO, cat_only: bool, camera_index: int = 0) -> None:
    """Tampilkan webcam dengan bounding box, nama kelas, confidence, jumlah objek, dan FPS."""
    import cv2

    print("Membuka webcam... Tekan Q atau ESC pada jendela video untuk berhenti.")
    print("Mode deteksi:", "kucing saja" if cat_only else "semua kelas COCO")

    # stream=True mengirim hasil per frame; result.plot() menggambar label dan bounding box.
    results = model.predict(
        source=camera_index,
        stream=True,
        show=False,
        verbose=False,
        conf=0.25,
        classes=prediction_classes(cat_only),
    )

    previous_time = time.perf_counter()
    fps = 0.0
    try:
        for result in results:
            # plot() menghasilkan frame BGR yang sudah diberi bounding box, nama kelas,
            # dan confidence. Saat tidak ada objek, frame kamera tetap ditampilkan.
            frame = result.plot(labels=True, conf=True)
            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (0.8 * fps + 0.2 * instant_fps)

            object_count = len(result.boxes) if result.boxes is not None else 0
            cv2.putText(
                frame, f"Objek terdeteksi: {object_count}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.imshow("YOLO Real-Time Object Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q atau ESC
                break
    finally:
        # Tutup generator agar kamera dilepas dan tutup semua jendela OpenCV.
        close = getattr(results, "close", None)
        if callable(close):
            close()
        cv2.destroyAllWindows()


def do_export(model: YOLO) -> None:
    exported_path = model.export(format="onnx", imgsz=640)
    print(f"Ekspor ONNX berhasil: {exported_path}")
    print(f"File tersedia: {Path(exported_path).exists()}")


def do_benchmark(model: YOLO, image_path: str, iterations: int) -> None:
    import cv2

    image = Path(image_path).expanduser()
    if not image.is_absolute():
        image = ROOT / image
    if not image.is_file():
        raise FileNotFoundError(f"Gambar benchmark tidak ditemukan: {image}")
    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {image}")

    process = psutil.Process(os.getpid())
    memory_before_mb = process.memory_info().rss / (1024 ** 2)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warm-up agar pengukuran tidak didominasi proses pemuatan awal model.
    for _ in range(3):
        model.predict(source=frame, verbose=False, conf=0.25)
    if cuda_available:
        torch.cuda.synchronize()

    durations_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict(source=frame, verbose=False, conf=0.25)
        if cuda_available:
            torch.cuda.synchronize()
        durations_ms.append((time.perf_counter() - start) * 1000)

    memory_after_mb = process.memory_info().rss / (1024 ** 2)
    avg_latency = statistics.mean(durations_ms)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    print("\nHASIL BENCHMARK")
    print(f"Perangkat          : {'CUDA GPU' if cuda_available else 'CPU'}")
    print(f"Jumlah pengulangan : {iterations}")
    print(f"Latency rata-rata  : {avg_latency:.2f} ms/gambar")
    print(f"Latency minimum   : {min(durations_ms):.2f} ms")
    print(f"Latency maksimum  : {max(durations_ms):.2f} ms")
    print(f"FPS perkiraan      : {fps:.2f}")
    print(f"Memori RAM sebelum : {memory_before_mb:.2f} MB")
    print(f"Memori RAM sesudah : {memory_after_mb:.2f} MB")
    print(f"Perubahan RAM      : {memory_after_mb - memory_before_mb:+.2f} MB")
    if cuda_available:
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak memori GPU    : {peak_gpu_mb:.2f} MB")
    print("Catatan: hasil benchmark bergantung pada perangkat, suhu, dan proses lain yang berjalan.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["val", "image", "video", "webcam", "export", "benchmark"],
        help="Tahap yang ingin dijalankan sesuai slide.",
    )
    parser.add_argument("--weights", help="Path file .pt; default yolo26n.pt")
    parser.add_argument("--source", help="Path gambar/video; untuk image/video")
    parser.add_argument("--data", default="coco8.yaml", help="Dataset validasi; default coco8.yaml")
    parser.add_argument("--cats-only", action="store_true", help="Batasi deteksi hanya pada kelas kucing")
    parser.add_argument("--iterations", type=int, default=30, help="Jumlah pengulangan benchmark")
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations harus minimal 1")

    model = load_model(args.weights)
    cat_only = args.cats_only

    if args.action == "val":
        do_validation(model, args.data)
    elif args.action == "image":
        do_predict(model, args.source or "sample.jpg", cat_only)
    elif args.action == "video":
        do_predict(model, args.source or "video.mp4", cat_only)
    elif args.action == "webcam":
        do_webcam(model, cat_only)
    elif args.action == "export":
        do_export(model)
    elif args.action == "benchmark":
        do_benchmark(model, args.source or "sample.jpg", args.iterations)


if __name__ == "__main__":
    main()
