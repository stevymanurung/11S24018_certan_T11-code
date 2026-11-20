
from ultralytics import YOLO

# Contoh model pretrained dari dokumentasi terbaru
model = YOLO("yolo26n.pt")

# Prediksi satu gambar
results = model.predict(
    source="sample.jpg",
    conf=0.25,
    save=True
)

# Baca hasil deteksi
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(model.names[cls_id], score, xyxy)
