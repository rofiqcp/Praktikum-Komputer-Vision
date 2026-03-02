"""
==========================================================================
PERCOBAAN 14: INSTANCE SEGMENTATION (KONSEP)
==========================================================================
Program ini mendemonstrasikan perbedaan semantic vs instance segmentation.
Instance segmentation tidak hanya melabeli kelas setiap piksel, tetapi
juga membedakan antar objek yang berbeda dari kelas yang sama.

Konsep yang dipelajari:
- Perbedaan: klasifikasi, deteksi, semantic seg, instance seg
- Mask per-instance untuk setiap objek
- Simulasi Mask R-CNN pipeline
- Evaluasi: mask IoU

Referensi: Szeliski Ch.5, Deep Learning for CV (Rosebrock)
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


def load_gambar(nama_file):
    """Memuat gambar dari folder image/."""
    path = os.path.join(IMAGE_DIR, nama_file)
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] {nama_file} tidak ditemukan!")
    return img


def buat_gambar_sintetis():
    """
    Membuat gambar sintetis dengan beberapa objek untuk demonstrasi.
    Objek dari kelas sama tetapi instance berbeda.
    """
    canvas = np.ones((300, 400, 3), dtype=np.uint8) * 240
    objects = []

    # Lingkaran (kelas "bola")
    cv2.circle(canvas, (80, 100), 40, (0, 0, 200), -1)
    objects.append({"kelas": "bola", "id": 1, "center": (80, 100), "radius": 40})
    cv2.circle(canvas, (200, 80), 35, (0, 0, 180), -1)
    objects.append({"kelas": "bola", "id": 2, "center": (200, 80), "radius": 35})

    # Persegi (kelas "kotak")
    cv2.rectangle(canvas, (280, 60), (360, 140), (200, 0, 0), -1)
    objects.append({"kelas": "kotak", "id": 3, "box": (280, 60, 360, 140)})
    cv2.rectangle(canvas, (100, 180), (180, 260), (180, 0, 0), -1)
    objects.append({"kelas": "kotak", "id": 4, "box": (100, 180, 180, 260)})

    # Segitiga (kelas "segitiga")
    pts = np.array([[300, 250], [350, 170], [250, 170]], np.int32)
    cv2.fillPoly(canvas, [pts], (0, 180, 0))
    objects.append({"kelas": "segitiga", "id": 5, "pts": pts})

    return canvas, objects


def buat_mask_instance(canvas_shape, objects):
    """
    Membuat mask per-instance untuk setiap objek.
    Setiap objek mendapat mask terpisah dengan ID unik.
    """
    masks = {}
    for obj in objects:
        mask = np.zeros(canvas_shape[:2], dtype=np.uint8)
        if "radius" in obj:
            cv2.circle(mask, obj["center"], obj["radius"], 255, -1)
        elif "box" in obj:
            x1, y1, x2, y2 = obj["box"]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        elif "pts" in obj:
            cv2.fillPoly(mask, [obj["pts"]], 255)
        masks[obj["id"]] = {"mask": mask, "kelas": obj["kelas"]}
    return masks


def visualisasi_perbandingan_task(canvas, masks):
    """
    Membandingkan 4 tugas vision: Klasifikasi, Deteksi, Semantic Seg, Instance Seg.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Klasifikasi
    axes[0, 0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Klasifikasi\n(Apa objeknya?)", fontsize=11)
    axes[0, 0].text(5, 295, "Kelas: bola, kotak, segitiga", fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='yellow'))

    # 2. Deteksi
    img_det = canvas.copy()
    colors_det = {'bola': (0, 0, 255), 'kotak': (255, 0, 0), 'segitiga': (0, 255, 0)}
    for mid, info in masks.items():
        mask = info["mask"]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            color = colors_det.get(info["kelas"], (128, 128, 128))
            cv2.rectangle(img_det, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_det, info["kelas"], (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    axes[0, 1].imshow(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Deteksi Objek\n(Di mana objeknya?)", fontsize=11)

    # 3. Semantic Segmentation
    semantic_map = np.zeros(canvas.shape[:2], dtype=np.uint8)
    kelas_map = {'bola': 1, 'kotak': 2, 'segitiga': 3}
    for mid, info in masks.items():
        semantic_map[info["mask"] > 0] = kelas_map.get(info["kelas"], 0)
    axes[1, 0].imshow(semantic_map, cmap='tab10')
    axes[1, 0].set_title("Semantic Segmentation\n(Label per piksel, tapi tanpa instance)", fontsize=11)

    # 4. Instance Segmentation
    instance_map = np.zeros(canvas.shape[:2], dtype=np.uint8)
    for mid, info in masks.items():
        instance_map[info["mask"] > 0] = mid
    axes[1, 1].imshow(instance_map, cmap='tab10')
    axes[1, 1].set_title("Instance Segmentation\n(Setiap objek terpisah)", fontsize=11)

    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle("Percobaan 14: Perbandingan Tugas Vision", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "14_instance_vs_semantic.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/14_instance_vs_semantic.png")


def visualisasi_mask_overlay(canvas, masks):
    """Menampilkan setiap instance mask dengan warna overlay berbeda."""
    overlay = canvas.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for i, (mid, info) in enumerate(masks.items()):
        color = colors[i % len(colors)]
        overlay[info["mask"] > 0] = color

    blended = cv2.addWeighted(canvas, 0.5, overlay, 0.5, 0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "14_instance_overlay.jpg"), blended)
    print("  [SAVED] output/14_instance_overlay.jpg")
    return blended


def main():
    """Fungsi utama: instance segmentation konsep."""
    print("=" * 60)
    print("PERCOBAAN 14: INSTANCE SEGMENTATION (KONSEP)")
    print("=" * 60)

    print("\n--- 1. Membuat Gambar Sintetis ---")
    canvas, objects = buat_gambar_sintetis()
    print(f"  Jumlah objek: {len(objects)}")
    for obj in objects:
        print(f"  - ID {obj['id']}: {obj['kelas']}")

    print("\n--- 2. Membuat Mask per-Instance ---")
    masks = buat_mask_instance(canvas.shape, objects)

    print("\n--- 3. Perbandingan 4 Tugas Vision ---")
    visualisasi_perbandingan_task(canvas, masks)

    print("\n--- 4. Instance Mask Overlay ---")
    blended = visualisasi_mask_overlay(canvas, masks)

    cv2.imshow("Instance Segmentation - Percobaan 14", blended)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 14")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. Klasifikasi: menentukan kelas gambar
2. Deteksi: menentukan lokasi + kelas (bounding box)
3. Semantic Segmentation: label per piksel (tanpa beda instance)
4. Instance Segmentation: label per piksel + beda instance
5. Mask R-CNN adalah arsitektur utama untuk instance segmentation

Output: output/14_instance_vs_semantic.png, output/14_instance_overlay.jpg
""")


if __name__ == "__main__":
    main()
