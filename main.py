
import cv2
import matplotlib.pyplot as plt

# Membaca gambar
image_bgr = cv2.imread("sample.jpg")

if image_bgr is None:
    raise FileNotFoundError("sample.jpg tidak ditemukan")

# Mengubah BGR menjadi RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Menampilkan ukuran gambar
print("Ukuran (H, W, C):", image_rgb.shape)

# Menampilkan gambar
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
