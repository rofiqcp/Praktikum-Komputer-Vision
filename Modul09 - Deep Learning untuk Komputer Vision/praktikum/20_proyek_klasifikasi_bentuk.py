"""
==========================================================================
PERCOBAAN 20: PROYEK KLASIFIKASI BENTUK
==========================================================================
Program ini adalah proyek akhir yang menggabungkan semua konsep dari
Modul 09. Membuat sistem klasifikasi bentuk geometris (lingkaran,
segitiga, persegi, bintang) menggunakan teknik computer vision.

Konsep yang diaplikasikan:
- Augmentasi data (Percobaan 7)
- Preprocessing dan normalisasi (Percobaan 1, 6)
- Evaluasi model (Percobaan 18)
- Visualisasi hasil klasifikasi
- Pipeline lengkap: generate dataset -> train -> evaluate -> test

Referensi: Szeliski Ch.5, semua percobaan Modul 09
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def buat_lingkaran(size=64, augment=False):
    """Membuat gambar bentuk lingkaran (putih di atas hitam)."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = size // 3
    if augment:
        cx += np.random.randint(-5, 6)
        cy += np.random.randint(-5, 6)
        r += np.random.randint(-3, 4)
    cv2.circle(img, (cx, cy), max(r, 5), 255, -1)
    return img


def buat_segitiga(size=64, augment=False):
    """Membuat gambar bentuk segitiga."""
    img = np.zeros((size, size), dtype=np.uint8)
    margin = size // 5
    pts = np.array([
        [size // 2, margin],
        [margin, size - margin],
        [size - margin, size - margin]
    ])
    if augment:
        pts += np.random.randint(-4, 5, pts.shape)
    cv2.fillPoly(img, [pts], 255)
    return img


def buat_persegi(size=64, augment=False):
    """Membuat gambar bentuk persegi/kotak."""
    img = np.zeros((size, size), dtype=np.uint8)
    margin = size // 5
    x1, y1 = margin, margin
    x2, y2 = size - margin, size - margin
    if augment:
        dx, dy = np.random.randint(-4, 5, 2)
        x1 += dx; y1 += dy; x2 += dx; y2 += dy
    cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
    return img


def buat_bintang(size=64, augment=False):
    """Membuat gambar bentuk bintang 5 titik."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r_outer = size // 3
    r_inner = size // 6

    pts = []
    for i in range(10):
        angle = np.pi / 2 + i * np.pi / 5
        r = r_outer if i % 2 == 0 else r_inner
        x = int(cx + r * np.cos(angle))
        y = int(cy - r * np.sin(angle))
        pts.append([x, y])

    pts = np.array(pts)
    if augment:
        pts += np.random.randint(-3, 4, pts.shape)
    cv2.fillPoly(img, [pts], 255)
    return img


def generate_dataset(n_per_class=100, size=64):
    """
    Membuat dataset sintetis berisi 4 bentuk: lingkaran, segitiga, persegi, bintang.
    Masing-masing n_per_class sampel dengan augmentasi random.
    """
    generators = [buat_lingkaran, buat_segitiga, buat_persegi, buat_bintang]
    class_names = ["Lingkaran", "Segitiga", "Persegi", "Bintang"]

    images = []
    labels = []

    for cls_id, gen_func in enumerate(generators):
        for i in range(n_per_class):
            augment = (i > 0)  # Augmentasi untuk semua kecuali sampel pertama
            img = gen_func(size=size, augment=augment)
            # Tambah noise
            if augment:
                noise = np.random.normal(0, 10, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            images.append(img)
            labels.append(cls_id)

    images = np.array(images)
    labels = np.array(labels)

    # Shuffle
    idx = np.random.permutation(len(images))
    return images[idx], labels[idx], class_names


def extract_features(images):
    """
    Ekstraksi fitur dari gambar menggunakan Hu Moments + kontur statistik.
    Fitur yang diekstrak:
    - 7 Hu Moments (invarian terhadap rotasi, skala, translasi)
    - Luas dan keliling kontur
    - Circularitas = 4*pi*area / perimeter^2
    """
    features = []
    for img in images:
        # Hu Moments
        moments = cv2.moments(img)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        # Kontur statistik
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)
        else:
            area, perimeter, circularity = 0, 0, 0

        feat = np.concatenate([hu, [area, perimeter, circularity]])
        features.append(feat)

    return np.array(features, dtype=np.float32)


def klasifikasi_knn(train_feat, train_labels, test_feat, k=5):
    """
    Klasifikasi menggunakan K-Nearest Neighbors sederhana.
    Menghitung jarak Euclidean ke semua sampel training,
    lalu voting berdasarkan k tetangga terdekat.
    """
    predictions = []
    for feat in test_feat:
        dists = np.sqrt(np.sum((train_feat - feat) ** 2, axis=1))
        nearest = np.argsort(dists)[:k]
        nearest_labels = train_labels[nearest]
        # Voting
        counts = np.bincount(nearest_labels, minlength=4)
        predictions.append(np.argmax(counts))
    return np.array(predictions)


def visualisasi_dataset(images, labels, class_names):
    """Menampilkan sampel dataset (4 sampel per kelas)."""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for cls_id in range(4):
        idx = np.where(labels == cls_id)[0][:4]
        for j, i in enumerate(idx):
            axes[cls_id, j].imshow(images[i], cmap='gray')
            axes[cls_id, j].axis('off')
            if j == 0:
                axes[cls_id, j].set_ylabel(class_names[cls_id], fontsize=10)
    plt.suptitle("Sampel Dataset Klasifikasi Bentuk", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "20_dataset_samples.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/20_dataset_samples.png")


def visualisasi_confusion_matrix(cm, classes):
    """Menampilkan confusion matrix sebagai heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=14)
    ax.set_ylabel("Label Sebenarnya")
    ax.set_xlabel("Label Prediksi")
    ax.set_title("Confusion Matrix - Klasifikasi Bentuk")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "20_confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/20_confusion_matrix.png")


def visualisasi_prediksi(images, labels, predictions, class_names):
    """Menampilkan hasil prediksi dengan warna hijau (benar) / merah (salah)."""
    n_show = 20
    idx = np.random.choice(len(images), min(n_show, len(images)), replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(idx):
            j = idx[i]
            ax.imshow(images[j], cmap='gray')
            correct = labels[j] == predictions[j]
            color = 'green' if correct else 'red'
            ax.set_title(f"True: {class_names[labels[j]]}\n"
                         f"Pred: {class_names[predictions[j]]}",
                         color=color, fontsize=8)
        ax.axis('off')
    plt.suptitle("Hasil Prediksi (Hijau=Benar, Merah=Salah)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "20_prediksi_hasil.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/20_prediksi_hasil.png")


def main():
    """Fungsi utama: proyek klasifikasi bentuk end-to-end."""
    print("=" * 60)
    print("PERCOBAAN 20: PROYEK KLASIFIKASI BENTUK")
    print("=" * 60)

    np.random.seed(42)

    # --- 1. Generate Dataset ---
    print("\n--- 1. Generate Dataset ---")
    images, labels, class_names = generate_dataset(n_per_class=100)
    print(f"  Total dataset: {len(images)} gambar")
    print(f"  Kelas: {class_names}")
    print(f"  Distribusi: {[np.sum(labels == i) for i in range(4)]}")
    visualisasi_dataset(images, labels, class_names)

    # --- 2. Split Train/Test ---
    print("\n--- 2. Split Train/Test (80/20) ---")
    n = len(images)
    n_train = int(0.8 * n)
    train_img, test_img = images[:n_train], images[n_train:]
    train_lbl, test_lbl = labels[:n_train], labels[n_train:]
    print(f"  Training: {len(train_img)} gambar")
    print(f"  Testing:  {len(test_img)} gambar")

    # --- 3. Ekstraksi Fitur ---
    print("\n--- 3. Ekstraksi Fitur (Hu Moments + Kontur) ---")
    train_feat = extract_features(train_img)
    test_feat = extract_features(test_img)
    print(f"  Dimensi fitur: {train_feat.shape[1]}")

    # Normalisasi fitur
    mean = train_feat.mean(axis=0)
    std = train_feat.std(axis=0) + 1e-8
    train_feat = (train_feat - mean) / std
    test_feat = (test_feat - mean) / std

    # --- 4. Klasifikasi KNN ---
    print("\n--- 4. Klasifikasi KNN (k=5) ---")
    predictions = klasifikasi_knn(train_feat, train_lbl, test_feat, k=5)
    accuracy = np.mean(predictions == test_lbl)
    print(f"  Akurasi: {accuracy:.2%}")

    # --- 5. Evaluasi ---
    print("\n--- 5. Evaluasi ---")
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(test_lbl, predictions):
        cm[t][p] += 1
    visualisasi_confusion_matrix(cm, class_names)

    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {cls}: Precision={prec:.2f}, Recall={rec:.2f}")

    # --- 6. Visualisasi Prediksi ---
    print("\n--- 6. Visualisasi Prediksi ---")
    visualisasi_prediksi(test_img, test_lbl, predictions, class_names)

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 20 (PROYEK)")
    print("=" * 60)
    print(f"""
Pipeline Klasifikasi Bentuk Geometris:
1. Generate dataset sintetis: 4 kelas x 100 sampel
2. Augmentasi: posisi random, noise Gaussian
3. Fitur: Hu Moments (7) + Luas + Keliling + Circularitas (10 fitur)
4. Normalisasi: z-score (mean=0, std=1)
5. Klasifikasi: KNN (k=5)
6. Hasil: Akurasi = {accuracy:.2%}

Output: output/20_dataset_samples.png, output/20_confusion_matrix.png,
        output/20_prediksi_hasil.png
""")


if __name__ == "__main__":
    main()
