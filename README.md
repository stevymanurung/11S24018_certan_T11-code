# Praktikum Computer Vision — Slide Validasi, Inferensi, dan Ekspor YOLO

Proyek ini disiapkan untuk mempraktikkan isi slide: **validasi model, inferensi gambar/video/webcam, ekspor ONNX, dan benchmark latency/FPS/memori**.

## Model yang digunakan

- `yolo26n.pt` adalah model YOLO pretrained pada dataset COCO dan sudah disertakan dalam folder proyek.
- Model ini bukan model khusus kucing. Model dapat mengenali beberapa kelas COCO, termasuk `cat` (ID kelas 15).
- Secara default, demo inferensi menampilkan semua kelas COCO. Tambahkan `--cats-only` jika ingin membatasi deteksi hanya pada kucing.
- `sample.jpg` di folder proyek adalah gambar kucing yang dapat langsung dipakai untuk uji inferensi.

> Untuk latihan slide ini, kamu tidak perlu menjalankan training melon terlebih dahulu. File `best.pt` dari training melon berbeda dengan model pretrained COCO yang dipakai pada demo kucing.

## 1. Instalasi

Buka terminal VS Code pada folder `belajar-computer-vision`:

```powershell
python -m pip install -r requirements.txt
```

## 2. Validasi model

```powershell
python slide26_demo.py val
```

Perintah ini menjalankan `model.val(data="coco8.yaml")`. Ultralytics akan mengunduh dataset contoh COCO8 saat pertama kali digunakan jika belum tersedia. Hasil validasi menampilkan Precision, Recall, mAP50, dan mAP50-95. Dataset COCO8 merupakan dataset contoh kecil untuk menguji alur validasi, bukan evaluasi ilmiah performa deteksi kucing.

Jika ingin memakai dataset validasi lain yang sesuai dengan kelas COCO:

```powershell
python slide26_demo.py val --data "C:/lokasi/dataset/data.yaml"
```

Jangan menggunakan `dataset/data.yaml` melon untuk memvalidasi model COCO pretrained karena label kelasnya berbeda.

## 3. Inferensi gambar

```powershell
python slide26_demo.py image --source sample.jpg
```

Hasil gambar disimpan otomatis di folder `runs/detect/`. Default menampilkan semua kelas COCO. Untuk membatasi ke kucing saja:

```powershell
python slide26_demo.py image --source sample.jpg --cats-only
```

## 4. Inferensi video

Letakkan video milik sendiri dengan nama `video.mp4` di folder utama proyek, lalu jalankan:

```powershell
python slide26_demo.py video --source video.mp4
```

Secara default semua kelas COCO ditampilkan. Untuk kucing saja, tambahkan `--cats-only`. Atau tentukan lokasi video:

```powershell
python slide26_demo.py video --source "D:/Video/kucing.mp4"
```

Hasil video akan disimpan di folder `runs/detect/`. Video tidak disertakan karena pengguna perlu memilih video yang dimiliki/diizinkan untuk digunakan.

## 5. Inferensi webcam

```powershell
python slide26_demo.py webcam
```

Webcam menampilkan bounding box, nama kelas, confidence, jumlah objek, dan FPS. Secara default semua kelas COCO dicoba dideteksi. Tekan `Q` atau `ESC` pada jendela kamera untuk berhenti. Untuk kucing saja, jalankan `python slide26_demo.py webcam --cats-only`. Pastikan kamera tidak sedang dipakai aplikasi lain.

## 6. Ekspor model ONNX

```powershell
python slide26_demo.py export
```

Jika berhasil, terminal menampilkan lokasi file `.onnx` dan status keberadaan file tersebut.

## 7. Benchmark perangkat

```powershell
python slide26_demo.py benchmark
```

Benchmark melakukan 3 warm-up, lalu mengukur 30 inferensi pada `sample.jpg`. Hasil menampilkan latency rata-rata/minimum/maksimum, FPS perkiraan, dan penggunaan RAM proses sebelum/sesudah. Jika CUDA tersedia, peak memori GPU juga ditampilkan. Atur jumlah pengulangan, misalnya:

```powershell
python slide26_demo.py benchmark --iterations 50
```

Pengukuran dipengaruhi spesifikasi perangkat, beban komputer, dan kondisi runtime. Catat hasil aktual dari laptop saat praktikum.

## 8. Perintah lengkap untuk bukti praktikum

Jalankan satu per satu dan screenshot hasil setiap tahap:

```powershell
python slide26_demo.py val
python slide26_demo.py image --source sample.jpg
python slide26_demo.py video --source video.mp4
python slide26_demo.py webcam
python slide26_demo.py export
python slide26_demo.py benchmark
```

Untuk tahap video, siapkan `video.mp4` terlebih dahulu. Untuk webcam, siapkan kamera laptop.

## Catatan file lama

- `train.py` dan `dataset/` tetap disertakan sebagai latihan terpisah untuk model custom melon.
- `validasi_inferensi_ekspor.py` adalah skrip versi sebelumnya untuk model custom melon. Untuk mengikuti slide menggunakan model pretrained/kucing, gunakan **`slide26_demo.py`**.
- Jangan mencampur `best.pt` model melon dengan `coco8.yaml` atau menganggap hasil model pretrained sebagai hasil training custom.
