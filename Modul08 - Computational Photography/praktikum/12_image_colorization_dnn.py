"""
==========================================================================
PERCOBAAN 12: IMAGE COLORIZATION (GRAYSCALE KE WARNA)
==========================================================================
Program ini mempelajari image colorization (grayscale ke warna).
Praktikum 12 - Image Colorization (Grayscale ke Warna)
Modul 08: Computational Photography

Topik: DNN-based colorization (Zhang 2016), Lab color space, ab prediction
Referensi: Mastering OpenCV 4 with Python Ch.12,
           OpenCV DNN module documentation

Hasil: Visualisasi dan analisis disimpan ke folder output/
==========================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)



def explain_colorization_theory():
    """Penjelasan konsep image colorization berbasis deep learning."""
    print("=" * 60)
    print("KONSEP IMAGE COLORIZATION")
    print("=" * 60)
    print("""
Colorization otomatis mengubah gambar grayscale menjadi gambar berwarna
menggunakan deep learning.

Pipeline (Zhang et al., 2016 — "Colorful Image Colorization"):
  ┌────────────────────────────────────────────────────────┐
  │  Grayscale (L channel dari Lab)                        │
  │       │                                                │
  │       ▼                                                │
  │  CNN (VGG-like encoder-decoder)                        │
  │       │                                                │
  │       ▼                                                │
  │  Prediksi ab channels (2 channel warna)                │
  │       │                                                │
  │       ▼                                                │
  │  Gabungkan L + ab → Lab → BGR                          │
  └────────────────────────────────────────────────────────┘

Kenapa Lab?
  - L channel = luminance (grayscale aslinya)
  - ab channel = komponen warna (bisa diprediksi dari L)
  - Lebih alami untuk task colorization

Model: colorization_release_v2.caffemodel
Tersedia di: https://github.com/richzhang/colorization
""")


def demo_lab_color_space():
    """Visualisasi ruang warna Lab dan pemisahan L, a, b channel."""
    # Buat gambar berwarna sintetis
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (200, 200), (0, 0, 200), -1)     # merah
    cv2.rectangle(img, (200, 0), (400, 200), (60, 180, 60), -1)  # hijau

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    plt.figure(figsize=(14, 4))
    panels = [
        ("BGR Asli", img, False),
        ("L channel\n(Luminance/Grayscale)", L, True),
        ("a channel\n(Hijau↔Merah)", a, True),
        ("b channel\n(Biru↔Kuning)", b, True),
    ]
    for i, (title, ch, gray) in enumerate(panels):
        plt.subplot(1, 4, i + 1)
        if gray:
            plt.imshow(ch, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(ch, cv2.COLOR_BGR2RGB))
        plt.title(title); plt.axis('off')
    plt.suptitle("Dekomposisi Ruang Warna Lab — Kunci Image Colorization")
    plt.tight_layout(); plt.savefig("output_12_lab_space.png", dpi=100); plt.show()

    print(f"  L range: {L.min()}-{L.max()} (luminance)")
    print(f"  a range: {a.min()}-{a.max()} (hijau-merah)")
    print(f"  b range: {b.min()}-{b.max()} (biru-kuning)")


def demo_simple_colorization_rule_based():
    """Colorization sederhana berbasis rules (sebelum DL era)."""
    # Gambar grayscale sintetis
    gray = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(gray, (130, 150), 80, 200, -1)     # objek terang (misal: matahari)
    for y in range(200, 300):
        gray[y, :] = int(80 + (y - 200) * 0.5)   # tanah gelap
    gray[:130, :] = np.linspace(180, 220, 400).astype(np.uint8)  # langit terang

    # Rule-based: threshold intensity ke warna
    colored = np.zeros((*gray.shape, 3), dtype=np.uint8)
    # Langit (terang atas): biru
    sky_mask = (gray[:130, :] > 150)
    colored[:130, :, 0][sky_mask] = 180  # B
    colored[:130, :, 1][sky_mask] = 120  # G
    colored[:130, :, 2][sky_mask] = 50   # R
    # Matahari/objek bulat terang: kuning-oranye
    colored[gray > 180] = [0, 200, 220]  # BGR: oranye-kuning
    # Tanah: hijau gelap
    colored[200:, :] = [50, 120, 60]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(gray, cmap='gray'); plt.title("Grayscale Input"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)); plt.title("Rule-Based Colorization\n(Sederhana, tidak realistis)"); plt.axis('off')
    plt.tight_layout(); plt.savefig("output_12_rule_based_color.png", dpi=100); plt.show()
    print("[Rule-based] Hasil tidak natural — butuh deep learning untuk realistis")


def demo_dnn_colorization_pipeline():
    """Pipeline colorization dengan OpenCV DNN (jika model tersedia)."""
    import os

    MODEL_PATH = "colorization_release_v2.caffemodel"
    PROTOTXT_PATH = "colorization_deploy_v2.prototxt"
    POINTS_PATH = "pts_in_hull.npy"

    if all(os.path.isfile(f) for f in [MODEL_PATH, PROTOTXT_PATH, POINTS_PATH]):
        print("[DNN Colorization] Model tersedia, menjalankan inference...")

        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        pts = np.load(POINTS_PATH).transpose().reshape(2, 313, 1, 1).astype(np.float32)

        class8 = net.getLayerId("class8_ab")
        conv8  = net.getLayerId("conv8_313_rh")
        net.getLayer(class8).blobs = [pts]
        net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype=np.float32)]

        # Buat gambar grayscale test
        img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(img_bgr, (128, 128), 80, (150, 150, 150), -1)
        cv2.rectangle(img_bgr, (60, 180), (196, 256), (80, 80, 80), -1)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
        L = img_lab[:, :, 0] - 50

        blob = cv2.dnn.blobFromImage(L.reshape(1, 1, *L.shape))
        net.setInput(blob)
        ab_dec = net.forward()[0, :, :, :].transpose(1, 2, 0)
        ab_dec = cv2.resize(ab_dec, (256, 256))

        img_lab_out = np.zeros_like(img_lab)
        img_lab_out[:, :, 0] = img_lab[:, :, 0]
        img_lab_out[:, :, 1:] = ab_dec * 128 + 128
        img_lab_out = np.clip(img_lab_out, 0, 255).astype(np.uint8)
        colorized = cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.imshow(gray, cmap='gray'); plt.title("Grayscale Input"); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)); plt.title("DNN Colorized Output"); plt.axis('off')
        plt.suptitle("Image Colorization dengan DNN (Zhang 2016)")
        plt.tight_layout(); plt.savefig("output_12_dnn_colorized.png", dpi=100); plt.show()
    else:
        print(f"[INFO] Model DNN tidak ditemukan di direktori ini.")
        print(f"  Download dari: https://github.com/richzhang/colorization")
        print(f"  File dibutuhkan: {MODEL_PATH}, {PROTOTXT_PATH}, {POINTS_PATH}")
        demo_colorization_histogram_transfer()


def demo_colorization_histogram_transfer():
    """Color transfer sederhana: pindahkan statistik warna dari referensi."""
    # Source (grayscale dikira sebagai foto BW lama)
    src_gray = np.zeros((300, 400, 3), dtype=np.uint8)
    for y in range(150):
        src_gray[y, :] = int(150 + y * 0.5)
    cv2.circle(src_gray, (200, 120), 60, (200, 200, 200), -1)
    src_gray[200:] = 80

    # Reference (gambar berwarna dengan palette yang diinginkan)
    ref = np.zeros((300, 400, 3), dtype=np.uint8)
    for y in range(150):
        ref[y, :] = [int(180 + y * 0.5), int(100 + y * 0.3), 50]  # langit
    cv2.circle(ref, (200, 120), 60, (0, 200, 240), -1)  # matahari kuning
    ref[200:] = [50, 120, 50]  # tanah hijau

    # Histogram matching per channel di Lab space
    def match_histograms_lab(source, reference):
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2Lab).astype(np.float32)
        result_lab = np.zeros_like(src_lab)
        for c in range(3):
            src_mean, src_std = src_lab[:, :, c].mean(), src_lab[:, :, c].std()
            ref_mean, ref_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std()
            result_lab[:, :, c] = (src_lab[:, :, c] - src_mean) * (ref_std / (src_std + 1e-9)) + ref_mean
        return cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

    transferred = match_histograms_lab(src_gray, ref)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(src_gray, cv2.COLOR_BGR2RGB)); plt.title("Source (Grayscale-like)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)); plt.title("Reference (Palette Warna)"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(transferred, cv2.COLOR_BGR2RGB)); plt.title("Color Transfer\n(Lab Statistics Matching)"); plt.axis('off')
    plt.suptitle("Color Transfer — Alternatif Colorization Sederhana")
    plt.tight_layout(); plt.savefig("output_12_color_transfer.png", dpi=100); plt.show()
    print("[OK] Color transfer via Lab statistics matching berhasil")


if __name__ == "__main__":
    print("=" * 58)
    print("PRAKTIKUM 12: IMAGE COLORIZATION (GRAYSCALE → WARNA)")
    print("=" * 58)

    explain_colorization_theory()

    print("\n[1] Lab Color Space — Fondasi Colorization")
    demo_lab_color_space()

    print("\n[2] Rule-Based Colorization (Pra-DL Era)")
    demo_simple_colorization_rule_based()

    print("\n[3] DNN Colorization Pipeline (Zhang 2016)")
    demo_dnn_colorization_pipeline()

    print("\n[SELESAI] Semua demo image colorization berhasil dijalankan.")
