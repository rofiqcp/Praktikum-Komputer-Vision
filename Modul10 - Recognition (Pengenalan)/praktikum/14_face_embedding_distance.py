"""
==========================================================================
PERCOBAAN 14: FACE EMBEDDING DAN DISTANCE
==========================================================================
Konsep face embedding: representasi wajah sebagai vektor, perbandingan jarak.

Referensi: Mastering OpenCV 4
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


def hitung_embedding_sederhana(gray_face):
    """
    Simulasi face embedding: mengubah gambar wajah menjadi vektor fitur.
    Dalam praktik nyata: FaceNet/ArcFace menghasilkan vektor 128/512-d.
    Di sini: gunakan LBP histogram + resize sebagai embedding sederhana.
    """
    face = cv2.resize(gray_face, (64, 64))
    # PCA-like: resize ke vektor kecil
    small = cv2.resize(face, (8, 8)).flatten().astype(np.float32)
    small = small / (np.linalg.norm(small) + 1e-8)  # L2 normalize
    return small


def hitung_jarak(emb1, emb2, metric="euclidean"):
    """
    Menghitung jarak antara dua embedding.
    Euclidean: jarak L2 (semakin kecil = semakin mirip).
    Cosine: 1 - cos_similarity (semakin kecil = semakin mirip).
    """
    if metric == "euclidean":
        return np.sqrt(np.sum((emb1 - emb2)**2))
    elif metric == "cosine":
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        return 1 - sim
    return 0


def main():
    """Fungsi utama: face embedding dan distance."""
    print("=" * 60)
    print("PERCOBAAN 14: FACE EMBEDDING DAN DISTANCE")
    print("=" * 60)
    
    # Buat wajah sintetis
    np.random.seed(42)
    faces_dir = os.path.join(IMAGE_DIR, "faces")
    face_images = []
    face_names = []
    
    if os.path.exists(faces_dir):
        for person in sorted(os.listdir(faces_dir))[:3]:
            pdir = os.path.join(faces_dir, person)
            if os.path.isdir(pdir):
                for f in sorted(os.listdir(pdir))[:2]:
                    img = cv2.imread(os.path.join(pdir, f), 0)
                    if img is not None:
                        face_images.append(cv2.resize(img, (100, 100)))
                        face_names.append(person)
    
    if len(face_images) < 4:
        face_images, face_names = [], []
        for name in ["Andi", "Andi", "Budi", "Budi", "Citra", "Citra"]:
            base = np.random.randint(80, 200, (100, 100), dtype=np.uint8)
            idx = ["Andi","Budi","Citra"].index(name)
            cv2.circle(base, (50, 40), 20+idx*5, 180, -1)
            noise = np.random.normal(0, 10, base.shape)
            face_images.append(np.clip(base+noise, 0, 255).astype(np.uint8))
            face_names.append(name)
    
    print(f"\n  Wajah: {len(face_images)} ({face_names})")
    
    print("\n--- 1. Hitung Embedding ---")
    embeddings = [hitung_embedding_sederhana(f) for f in face_images]
    print(f"  Dimensi embedding: {len(embeddings[0])}")
    
    print("\n--- 2. Distance Matrix ---")
    n = len(face_images)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = hitung_jarak(embeddings[i], embeddings[j])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    im = ax1.imshow(dist_matrix, cmap='RdYlGn_r')
    ax1.set_xticks(range(n)); ax1.set_xticklabels(face_names, rotation=45)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(face_names)
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{dist_matrix[i,j]:.2f}", ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax1)
    ax1.set_title("Distance Matrix (Euclidean)")
    
    # Tampilkan wajah
    for i in range(min(n, 6)):
        ax_small = fig.add_axes([0.55 + (i%3)*0.14, 0.55 - (i//3)*0.35, 0.12, 0.25])
        ax_small.imshow(face_images[i], cmap='gray')
        ax_small.set_title(face_names[i], fontsize=8)
        ax_small.axis('off')
    ax2.axis('off')
    
    plt.suptitle("Face Embedding Distance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "14_embedding_distance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/14_embedding_distance.png")
    
    cv2.imshow("Face Embeddings", np.hstack(face_images[:4]))
    cv2.waitKey(0); cv2.destroyAllWindows()
    print("\nRINGKASAN: Face embedding merepresentasikan wajah sebagai vektor.")
    print("Jarak kecil = wajah sama, jarak besar = wajah berbeda.")


if __name__ == "__main__":
    main()
