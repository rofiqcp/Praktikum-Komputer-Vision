"""
==========================================================================
PERCOBAAN 1: LOADING DAN MENAMPILKAN GAMBAR
==========================================================================
Program ini mempelajari cara memuat (load) gambar dari file menggunakan
OpenCV dan menampilkannya di jendela GUI atau menyimpan hasilnya.

Fungsi utama yang dipelajari:
- cv2.imread()    : Membaca/memuat gambar dari file
- cv2.imshow()    : Menampilkan gambar di jendela GUI
- cv2.waitKey()   : Menunggu input keyboard
- cv2.destroyAllWindows() : Menutup semua jendela GUI

Catatan: Jika jendela GUI tidak tersedia (server/remote), gunakan
matplotlib atau simpan hasil langsung ke file.
==========================================================================
"""

# Mengimpor library OpenCV untuk pemrosesan gambar
import cv2

# Mengimpor library NumPy untuk operasi array
import numpy as np

# Mengimpor library os untuk operasi path file
import os

# Mengimpor matplotlib untuk alternatif menampilkan gambar
import matplotlib.pyplot as plt

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input dan output
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Membaca gambar dalam mode warna (BGR)
# cv2.imread(path, flag) - flag default = cv2.IMREAD_COLOR
# ============================================================
print("=" * 60)
print("PERCOBAAN 1: LOADING DAN MENAMPILKAN GAMBAR")
print("=" * 60)

# Membaca gambar kucing dalam mode warna penuh (3 channel: Blue, Green, Red)
# cv2.IMREAD_COLOR (1) = baca sebagai gambar berwarna, abaikan transparansi
img_color = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"), cv2.IMREAD_COLOR)

# Memeriksa apakah gambar berhasil dimuat (tidak None)
if img_color is None:
    print("[ERROR] Gambar tidak ditemukan! Jalankan download_image.py terlebih dahulu.")
    exit()

# Menampilkan informasi bahwa gambar berhasil dimuat
print(f"[INFO] Gambar berwarna berhasil dimuat.")
print(f"  - Dimensi: {img_color.shape}")  # (height, width, channels)
print(f"  - Tipe data: {img_color.dtype}")  # uint8 (0-255)

# ============================================================
# 2. Membaca gambar dalam mode grayscale (1 channel)
# cv2.IMREAD_GRAYSCALE (0) = konversi ke abu-abu saat loading
# ============================================================

# Membaca gambar yang sama dalam mode grayscale (abu-abu)
img_gray = cv2.imread(os.path.join(IMAGE_DIR, "kucing.jpg"), cv2.IMREAD_GRAYSCALE)

# Menampilkan informasi gambar grayscale
print(f"\n[INFO] Gambar grayscale berhasil dimuat.")
print(f"  - Dimensi: {img_gray.shape}")  # (height, width) - tanpa channel
print(f"  - Tipe data: {img_gray.dtype}")

# ============================================================
# 3. Membaca gambar dengan alpha channel (transparansi)
# cv2.IMREAD_UNCHANGED (-1) = baca apa adanya termasuk alpha
# ============================================================

# Membaca gambar PNG yang mungkin memiliki alpha channel
img_unchanged = cv2.imread(os.path.join(IMAGE_DIR, "warna_warni.png"), cv2.IMREAD_UNCHANGED)

# Menampilkan informasi gambar unchanged
print(f"\n[INFO] Gambar unchanged berhasil dimuat.")
print(f"  - Dimensi: {img_unchanged.shape}")
print(f"  - Tipe data: {img_unchanged.dtype}")

# ============================================================
# 4. Menampilkan gambar menggunakan matplotlib (lebih portable)
# Catatan: OpenCV menggunakan BGR, matplotlib menggunakan RGB
# ============================================================

# Membuat figure dengan 3 subplot untuk menampilkan 3 versi gambar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Gambar berwarna (konversi BGR -> RGB untuk matplotlib)
# cv2.cvtColor() mengkonversi ruang warna gambar
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
axes[0].imshow(img_rgb)
axes[0].set_title("Gambar Berwarna (RGB)")
# Menghilangkan sumbu/axis agar tampilan lebih bersih
axes[0].axis("off")

# Subplot 2: Gambar grayscale (gunakan colormap 'gray')
axes[1].imshow(img_gray, cmap="gray")
axes[1].set_title("Gambar Grayscale")
axes[1].axis("off")

# Subplot 3: Gambar unchanged
img_unch_rgb = cv2.cvtColor(img_unchanged, cv2.COLOR_BGR2RGB)
axes[2].imshow(img_unch_rgb)
axes[2].set_title("Gambar Unchanged")
axes[2].axis("off")

# Mengatur layout agar tidak saling tumpang tindih
plt.suptitle("Percobaan 1: Tiga Mode Pembacaan Gambar", fontsize=14, fontweight="bold")
plt.tight_layout()

# Menyimpan hasil visualisasi ke folder output
output_path = os.path.join(OUTPUT_DIR, "01_loading_gambar_hasil.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] Hasil disimpan di: {output_path}")

# Menampilkan gambar (opsional, bisa dikomentari jika tanpa GUI)
# plt.show()

# ============================================================
# 5. Menampilkan menggunakan cv2.imshow() (GUI mode)
# ============================================================

# Menampilkan gambar di jendela OpenCV (hanya bekerja jika ada GUI)
try:
    # cv2.imshow(nama_jendela, gambar) - menampilkan gambar di jendela
    cv2.imshow("Gambar Berwarna", img_color)
    cv2.imshow("Gambar Grayscale", img_gray)

    # cv2.waitKey(0) - menunggu sampai user menekan tombol apapun
    # Parameter 0 = tunggu tanpa batas waktu
    # Parameter 1000 = tunggu 1000ms (1 detik)
    print("\n[INFO] Tekan tombol apapun pada jendela gambar untuk menutup...")
    cv2.waitKey(3000)  # Tunggu 3 detik atau sampai tombol ditekan

    # cv2.destroyAllWindows() - menutup semua jendela yang dibuat oleh OpenCV
    cv2.destroyAllWindows()
except Exception as e:
    print(f"[INFO] GUI tidak tersedia: {e}")
    print("[INFO] Hasil sudah disimpan ke folder output/")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 1")
print("=" * 60)
print("Fungsi yang dipelajari:")
print("  1. cv2.imread(path, flag)  → Membaca gambar dari file")
print("     - IMREAD_COLOR (1)      → Baca sebagai BGR 3 channel")
print("     - IMREAD_GRAYSCALE (0)  → Baca sebagai grayscale 1 channel")
print("     - IMREAD_UNCHANGED (-1) → Baca apa adanya (termasuk alpha)")
print("  2. cv2.imshow(nama, img)   → Tampilkan gambar di jendela GUI")
print("  3. cv2.waitKey(ms)         → Tunggu input keyboard")
print("  4. cv2.destroyAllWindows() → Tutup semua jendela GUI")
print("  5. cv2.cvtColor(img, code) → Konversi ruang warna (BGR↔RGB)")
print("=" * 60)
