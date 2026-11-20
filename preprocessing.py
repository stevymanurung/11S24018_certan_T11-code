
import cv2
import matplotlib.pyplot as plt

# Membaca gambar
image = cv2.imread("sample.jpg")

if image is None:
    raise FileNotFoundError("sample.jpg tidak ditemukan")

# Mengubah BGR menjadi RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Resize
resized = cv2.resize(image_rgb, (300, 300))

# 2. Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Canny Edge Detection
edges = cv2.Canny(blur, 100, 200)

# Menampilkan semua hasil
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Ukuran gambar asli:", image.shape)
print("Ukuran gambar resize:", resized.shape)
print("Preprocessing berhasil!")
