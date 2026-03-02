"""
==========================================================================
PERCOBAAN 4: FACE RECOGNITION DENGAN EIGENFACES (PCA)
==========================================================================
Program ini mendemonstrasikan pengenalan wajah menggunakan Eigenfaces,
yaitu metode berbasis Principal Component Analysis (PCA).

Konsep: Wajah diproyeksikan ke eigenspace berdimensi rendah.
Pengenalan dilakukan dengan membandingkan jarak di eigenspace.
Referensi: Turk & Pentland (1991), Learning OpenCV
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


def buat_dataset():
    """Membuat dataset wajah sintetis 3 orang × 5 variasi."""
    np.random.seed(42)
    images, labels, names = [], [], ["Andi", "Budi", "Citra"]
    faces_dir = os.path.join(IMAGE_DIR, "faces")
    if os.path.exists(faces_dir):
        persons = sorted([d for d in os.listdir(faces_dir)
                         if os.path.isdir(os.path.join(faces_dir, d))])[:3]
        for lbl, person in enumerate(persons):
            pdir = os.path.join(faces_dir, person)
            for f in sorted(os.listdir(pdir))[:5]:
                img = cv2.imread(os.path.join(pdir, f), 0)
                if img is not None:
                    images.append(cv2.resize(img, (100, 100)))
                    labels.append(lbl)
            if lbl < len(names):
                names[lbl] = person.capitalize()
    if len(images) < 6:
        images, labels = [], []
        for lbl in range(3):
            base = np.random.randint(80, 200, (100, 100), dtype=np.uint8)
            cv2.circle(base, (50, 40), 20 + lbl * 5, 180, -1)
            for i in range(5):
                v = base.copy().astype(float) + np.random.normal(0, 10, base.shape)
                images.append(np.clip(v, 0, 255).astype(np.uint8))
                labels.append(lbl)
    return images, np.array(labels), names


def hitung_eigenfaces(images, n_components=10):
    """
    Menghitung eigenfaces menggunakan PCA.
    1. Flatten setiap gambar menjadi vektor
    2. Hitung mean face
    3. SVD untuk mendapatkan eigenvectors (eigenfaces)
    """
    data = np.array([img.flatten().astype(np.float64) for img in images])
    mean_face = np.mean(data, axis=0)
    centered = data - mean_face
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenfaces = Vt[:n_components]
    return mean_face, eigenfaces, S


def visualisasi_eigenfaces(mean_face, eigenfaces, img_shape):
    """Menampilkan mean face dan top eigenfaces."""
    n = min(len(eigenfaces), 8)
    fig, axes = plt.subplots(1, n + 1, figsize=(14, 3))
    axes[0].imshow(mean_face.reshape(img_shape), cmap='gray')
    axes[0].set_title("Mean Face")
    axes[0].axis('off')
    for i in range(n):
        ef = eigenfaces[i].reshape(img_shape)
        axes[i + 1].imshow(ef, cmap='gray')
        axes[i + 1].set_title(f"EF-{i + 1}")
        axes[i + 1].axis('off')
    plt.suptitle("Mean Face dan Top Eigenfaces", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_eigenfaces.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/04_eigenfaces.png")


def rekonstruksi_wajah(images, mean_face, eigenfaces, idx=0):
    """Merekonstruksi wajah dari berbagai jumlah komponen."""
    face = images[idx].flatten().astype(np.float64)
    centered = face - mean_face
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    axes[0].imshow(images[idx], cmap='gray')
    axes[0].set_title("Asli")
    axes[0].axis('off')
    for i, n_comp in enumerate([1, 3, 5, 8, len(eigenfaces)]):
        n_comp = min(n_comp, len(eigenfaces))
        weights = eigenfaces[:n_comp] @ centered
        recon = mean_face + weights @ eigenfaces[:n_comp]
        recon = np.clip(recon, 0, 255).reshape(images[0].shape)
        axes[i + 1].imshow(recon, cmap='gray')
        axes[i + 1].set_title(f"{n_comp} komp.")
        axes[i + 1].axis('off')
    plt.suptitle("Rekonstruksi Wajah dari Eigenfaces", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_rekonstruksi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/04_rekonstruksi.png")


def main():
    """Fungsi utama: Eigenfaces recognition."""
    print("=" * 60)
    print("PERCOBAAN 4: FACE RECOGNITION DENGAN EIGENFACES")
    print("=" * 60)

    images, labels, names = buat_dataset()
    print(f"\n  Dataset: {len(images)} gambar, {len(names)} orang")

    print("\n--- 1. Hitung Eigenfaces ---")
    n_comp = min(10, len(images) - 1)
    mean_face, eigenfaces, S = hitung_eigenfaces(images, n_comp)
    variance = S ** 2 / np.sum(S ** 2)
    print(f"  Jumlah komponen: {n_comp}")
    print(f"  Variance explained (top-5): {variance[:5].round(3)}")
    visualisasi_eigenfaces(mean_face, eigenfaces, images[0].shape)

    print("\n--- 2. Rekonstruksi Wajah ---")
    rekonstruksi_wajah(images, mean_face, eigenfaces, idx=0)

    print("\n--- 3. Recognition ---")
    try:
        recognizer = cv2.face.EigenFaceRecognizer_create(num_components=n_comp)
        resized = [cv2.resize(img, (100, 100)) for img in images]
        n_train = int(0.8 * len(resized))
        recognizer.train(resized[:n_train], labels[:n_train])
        correct = sum(1 for i in range(n_train, len(resized))
                      if recognizer.predict(resized[i])[0] == labels[i])
        acc = correct / max(len(resized) - n_train, 1) * 100
        print(f"  Akurasi: {acc:.1f}%")
    except:
        print("  [INFO] cv2.face tidak tersedia, skip recognizer test")

    print("\n" + "=" * 60)
    print("RINGKASAN: Eigenfaces menggunakan PCA untuk mengurangi dimensi")
    print("gambar wajah. Eigenfaces pertama menangkap variasi terbesar.")
    print("=" * 60)


if __name__ == "__main__":
    main()
