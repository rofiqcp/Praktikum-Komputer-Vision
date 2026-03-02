"""
==========================================================================
PERCOBAAN 3: FACE RECOGNITION DENGAN LBPH
==========================================================================
Program ini mendemonstrasikan pengenalan wajah menggunakan LBPH
(Local Binary Patterns Histograms) — metode recognition bawaan OpenCV.

Konsep yang dipelajari:
- LBP (Local Binary Pattern): encoding tekstur lokal
- Histogram LBP per region
- cv2.face.LBPHFaceRecognizer
- Training, prediksi, dan confidence

Referensi: Learning OpenCV, Mastering OpenCV 4
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gambar(nama_file):
    """Memuat gambar dari folder image/."""
    path = os.path.join(IMAGE_DIR, nama_file)
    img = cv2.imread(path)
    if img is None:
        print(f"  [WARN] Gambar {nama_file} tidak ditemukan.")
    return img


def hitung_lbp_manual(gray, radius=1):
    """
    Menghitung Local Binary Pattern secara manual.
    Untuk setiap piksel, bandingkan dengan 8 tetangganya.
    Jika tetangga >= center, bit=1, else bit=0.
    Hasilnya: nilai desimal 0-255 per piksel.
    """
    h, w = gray.shape
    lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            center = gray[y, x]
            val = 0
            for i, (dy, dx) in enumerate(offsets):
                if gray[y + dy, x + dx] >= center:
                    val |= (1 << i)
            lbp[y - radius, x - radius] = val
    return lbp


def buat_dataset_sintetis():
    """
    Membuat dataset wajah sintetis (3 orang × 5 variasi).
    Menggunakan gambar dari folder faces/ jika tersedia.
    """
    faces_dir = os.path.join(IMAGE_DIR, "faces")
    images = []
    labels = []
    names = []

    if os.path.exists(faces_dir):
        persons = sorted([d for d in os.listdir(faces_dir)
                         if os.path.isdir(os.path.join(faces_dir, d))])[:3]
        for label, person in enumerate(persons):
            person_dir = os.path.join(faces_dir, person)
            for fname in sorted(os.listdir(person_dir))[:5]:
                fpath = os.path.join(person_dir, fname)
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (100, 100))
                    images.append(img)
                    labels.append(label)
            names.append(person.capitalize())
    
    # Jika data kurang, buat sintetis
    if len(images) < 6:
        images, labels, names = [], [], ["Andi", "Budi", "Citra"]
        np.random.seed(42)
        for label in range(3):
            base = np.random.randint(80, 200, (100, 100), dtype=np.uint8)
            cv2.circle(base, (50, 40), 25, int(150 + label * 30), -1)
            cv2.circle(base, (40, 35), 5, 50, -1)
            cv2.circle(base, (60, 35), 5, 50, -1)
            for i in range(5):
                variation = base.copy()
                noise = np.random.normal(0, 10 + i * 3, base.shape)
                variation = np.clip(variation.astype(float) + noise, 0, 255).astype(np.uint8)
                images.append(variation)
                labels.append(label)

    return images, np.array(labels), names


def train_dan_prediksi_lbph(images, labels, names):
    """
    Melatih LBPH recognizer dan menguji prediksi.
    LBPH membandingkan histogram LBP antara gambar query dan training.
    """
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8)
    except AttributeError:
        print("  [WARN] cv2.face tidak tersedia. Install: pip install opencv-contrib-python")
        return

    # Split train/test (80/20)
    n = len(images)
    idx = list(range(n))
    np.random.shuffle(idx)
    n_train = int(0.8 * n)
    train_imgs = [images[i] for i in idx[:n_train]]
    train_lbls = np.array([labels[i] for i in idx[:n_train]])
    test_imgs = [images[i] for i in idx[n_train:]]
    test_lbls = np.array([labels[i] for i in idx[n_train:]])

    # Training
    recognizer.train(train_imgs, train_lbls)
    print(f"  Training: {n_train} gambar")
    print(f"  Testing:  {len(test_imgs)} gambar")

    # Prediksi
    correct = 0
    results = []
    for img, true_label in zip(test_imgs, test_lbls):
        pred_label, confidence = recognizer.predict(img)
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        results.append((img, true_label, pred_label, confidence, is_correct))

    accuracy = correct / len(test_imgs) * 100 if test_imgs else 0
    print(f"  Akurasi: {accuracy:.1f}%")

    # Visualisasi prediksi
    n_show = min(len(results), 8)
    fig, axes = plt.subplots(2, (n_show + 1) // 2, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            img, true_l, pred_l, conf, correct = results[i]
            ax.imshow(img, cmap='gray')
            color = 'green' if correct else 'red'
            ax.set_title(f"True:{names[true_l]}\nPred:{names[pred_l]}\n({conf:.0f})",
                        color=color, fontsize=8)
        ax.axis('off')
    plt.suptitle(f"LBPH Face Recognition (Akurasi: {accuracy:.1f}%)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_lbph_prediksi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/03_lbph_prediksi.png")


def visualisasi_lbp(img_gray):
    """Menampilkan gambar asli vs LBP pattern."""
    lbp = hitung_lbp_manual(img_gray)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    ax1.imshow(img_gray, cmap='gray')
    ax1.set_title("Gambar Grayscale")
    ax1.axis('off')

    ax2.imshow(lbp, cmap='gray')
    ax2.set_title("LBP Pattern")
    ax2.axis('off')

    ax3.hist(lbp.ravel(), bins=256, color='steelblue', alpha=0.7)
    ax3.set_title("Histogram LBP")
    ax3.set_xlabel("LBP Value")

    plt.suptitle("Local Binary Pattern (LBP)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_lbp_pattern.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/03_lbp_pattern.png")


def main():
    """Fungsi utama: face recognition LBPH."""
    print("=" * 60)
    print("PERCOBAAN 3: FACE RECOGNITION DENGAN LBPH")
    print("=" * 60)

    print("\n--- 1. Visualisasi LBP ---")
    img = load_gambar("wajah_single.jpg")
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
    else:
        gray = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    visualisasi_lbp(gray)

    print("\n--- 2. LBPH Face Recognition ---")
    images, labels, names = buat_dataset_sintetis()
    print(f"  Dataset: {len(images)} gambar, {len(names)} orang: {names}")
    train_dan_prediksi_lbph(images, labels, names)

    print("\n" + "=" * 60)
    print("RINGKASAN: LBPH membandingkan tekstur lokal (LBP histogram)")
    print("antar gambar wajah. Robust terhadap perubahan pencahayaan,")
    print("namun sensitif terhadap perubahan pose dan ekspresi.")
    print("=" * 60)


if __name__ == "__main__":
    main()
