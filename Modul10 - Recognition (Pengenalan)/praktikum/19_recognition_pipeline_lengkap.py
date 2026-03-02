"""
==========================================================================
PERCOBAAN 19: RECOGNITION PIPELINE LENGKAP
==========================================================================
Pipeline recognition end-to-end: detect → align → extract → match.

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


def deteksi(img):
    """Step 1: Deteksi wajah."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    return faces, gray


def alignment(gray, face_rect):
    """Step 2: Crop dan resize wajah (alignment sederhana)."""
    x, y, w, h = face_rect
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (100, 100))
    face = cv2.equalizeHist(face)
    return face


def extract_embedding(face):
    """Step 3: Ekstrak fitur/embedding dari wajah."""
    small = cv2.resize(face, (8, 8)).flatten().astype(np.float32)
    return small / (np.linalg.norm(small) + 1e-8)


def match(query_emb, database, threshold=0.8):
    """Step 4: Cocokkan dengan database."""
    best_name, best_dist = "Unknown", float('inf')
    for name, emb in database:
        dist = np.sqrt(np.sum((query_emb - emb)**2))
        if dist < best_dist:
            best_dist = dist; best_name = name
    if best_dist > threshold:
        best_name = "Unknown"
    return best_name, best_dist


def main():
    """Fungsi utama: pipeline recognition lengkap."""
    print("=" * 60)
    print("PERCOBAAN 19: RECOGNITION PIPELINE LENGKAP")
    print("=" * 60)
    
    # Buat database dari gambar
    database = []
    faces_dir = os.path.join(IMAGE_DIR, "faces")
    if os.path.exists(faces_dir):
        for person in sorted(os.listdir(faces_dir))[:3]:
            pdir = os.path.join(faces_dir, person)
            if os.path.isdir(pdir):
                for f in sorted(os.listdir(pdir))[:3]:
                    img = cv2.imread(os.path.join(pdir, f), 0)
                    if img is not None:
                        face = cv2.resize(img, (100, 100))
                        emb = extract_embedding(face)
                        database.append((person, emb))
    
    if not database:
        np.random.seed(42)
        for name in ["Andi", "Budi", "Citra"]:
            for _ in range(3):
                face = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
                emb = extract_embedding(face)
                database.append((name, emb))
    
    print(f"\n  Database: {len(database)} entries")
    
    # Pipeline diagram
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    steps = [("1. DETECT\nHaar/DNN", "#4ECDC4"), ("2. ALIGN\nCrop+Resize", "#FF6B6B"),
             ("3. EMBED\nFeature Extract", "#45B7D1"), ("4. MATCH\nDatabase Search", "#96CEB4"),
             ("5. RESULT\nIdentity/Unknown", "#FFEAA7")]
    for i, (label, color) in enumerate(steps):
        x = 1 + i * 2.5
        rect = plt.Rectangle((x-0.8, 0.5), 1.8, 1.2, facecolor=color, edgecolor='black', lw=2)
        ax.add_patch(rect)
        ax.text(x+0.1, 1.1, label, ha='center', va='center', fontsize=9, fontweight='bold')
        if i < len(steps)-1:
            ax.annotate('', xy=(x+1.7, 1.1), xytext=(x+1.0, 1.1),
                       arrowprops=dict(arrowstyle='->', lw=2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 2.5)
    ax.set_title("Pipeline Recognition: Detect → Align → Embed → Match", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "19_pipeline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/19_pipeline.png")
    
    # Test recognition
    img = load_gambar("wajah_single.jpg")
    if img is not None:
        faces, gray = deteksi(img)
        for (x,y,w,h) in faces:
            face = alignment(gray, (x,y,w,h))
            emb = extract_embedding(face)
            name, dist = match(emb, database)
            print(f"  Wajah ({x},{y}): {name} (dist={dist:.3f})")
    
    print("\nRINGKASAN: Pipeline recognition: Detect → Align → Embed → Match.")


if __name__ == "__main__":
    main()
