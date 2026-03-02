"""
==========================================================================
PERCOBAAN 20: PROYEK RECOGNITION SISTEM
==========================================================================
Proyek akhir: sistem pengenalan wajah lengkap dengan database dan evaluasi.

Referensi: Semua referensi Modul 10
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


def buat_database_wajah():
    """Membuat database wajah dari folder image/faces/."""
    database = {}
    faces_dir = os.path.join(IMAGE_DIR, "faces")
    if os.path.exists(faces_dir):
        for person in sorted(os.listdir(faces_dir)):
            pdir = os.path.join(faces_dir, person)
            if os.path.isdir(pdir):
                database[person] = []
                for f in sorted(os.listdir(pdir)):
                    img = cv2.imread(os.path.join(pdir, f), 0)
                    if img is not None:
                        database[person].append(cv2.resize(img, (100, 100)))
    
    if not database:
        np.random.seed(42)
        for name in ["Andi", "Budi", "Citra"]:
            imgs = []
            for _ in range(5):
                img = np.random.randint(80, 200, (100, 100), dtype=np.uint8)
                imgs.append(img)
            database[name] = imgs
    return database


def train_recognizer(database):
    """Melatih recognizer dari database wajah."""
    images, labels, names = [], [], []
    for i, (name, imgs) in enumerate(database.items()):
        names.append(name)
        for img in imgs:
            images.append(img)
            labels.append(i)
    
    recognizer = None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(images, np.array(labels))
        print(f"  Recognizer trained: {len(images)} gambar, {len(names)} orang")
    except:
        print("  [WARN] cv2.face tidak tersedia")
    return recognizer, names, images, np.array(labels)


def evaluasi_sistem(recognizer, images, labels, names):
    """Evaluasi sistem recognition: accuracy, per-class metrics."""
    if recognizer is None:
        return
    
    correct = 0
    predictions = []
    for img, true_label in zip(images, labels):
        pred, conf = recognizer.predict(img)
        predictions.append(pred)
        if pred == true_label:
            correct += 1
    
    acc = correct / len(images) * 100
    n_cls = len(names)
    predictions = np.array(predictions)
    
    # Confusion matrix
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(labels, predictions):
        cm[t][p] += 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(cm, cmap='YlOrRd')
    axes[0].set_xticks(range(n_cls)); axes[0].set_xticklabels(names, rotation=30)
    axes[0].set_yticks(range(n_cls)); axes[0].set_yticklabels(names)
    for i in range(n_cls):
        for j in range(n_cls):
            axes[0].text(j, i, str(cm[i,j]), ha='center', va='center')
    axes[0].set_title(f"Confusion Matrix (Acc={acc:.1f}%)")
    plt.colorbar(im, ax=axes[0])
    
    # Per-class accuracy
    per_class = [cm[i,i]/max(cm[i,:].sum(),1)*100 for i in range(n_cls)]
    axes[1].bar(names, per_class, color='steelblue')
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Per-Class Accuracy")
    axes[1].set_ylim(0, 110)
    for i, v in enumerate(per_class):
        axes[1].text(i, v+2, f"{v:.0f}%", ha='center')
    
    plt.suptitle("Evaluasi Sistem Recognition", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "20_evaluasi_sistem.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Akurasi keseluruhan: {acc:.1f}%")
    print("  [SAVED] output/20_evaluasi_sistem.png")


def main():
    """Fungsi utama: proyek recognition sistem."""
    print("=" * 60)
    print("PERCOBAAN 20: PROYEK RECOGNITION SISTEM")
    print("=" * 60)
    
    print("\n--- 1. Buat Database ---")
    database = buat_database_wajah()
    print(f"  Database: {list(database.keys())}")
    for name, imgs in database.items():
        print(f"    {name}: {len(imgs)} gambar")
    
    print("\n--- 2. Train Recognizer ---")
    recognizer, names, images, labels = train_recognizer(database)
    
    print("\n--- 3. Evaluasi ---")
    evaluasi_sistem(recognizer, images, labels, names)
    
    # Tampilkan sampel database
    fig, axes = plt.subplots(len(database), 5, figsize=(12, 3*len(database)))
    if len(database) == 1:
        axes = axes.reshape(1, -1)
    for i, (name, imgs) in enumerate(database.items()):
        for j in range(min(5, len(imgs))):
            axes[i, j].imshow(imgs[j], cmap='gray')
            axes[i, j].set_title(name if j==0 else "", fontsize=9)
            axes[i, j].axis('off')
    plt.suptitle("Database Wajah", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "20_database.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/20_database.png")
    
    print("\n" + "=" * 60)
    print("PROYEK SELESAI: Sistem recognition dengan database,")
    print("training LBPH, dan evaluasi metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
