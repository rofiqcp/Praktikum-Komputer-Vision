"""
==========================================================================
PERCOBAAN 16: STEREO MATCHING REALTIME
==========================================================================
Implementasi stereo matching real-time dengan parameter tuning.

Referensi: Learning OpenCV
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
        print(f"  [WARN] Gambar {nama_file} tidak ditemukan, gunakan sintetis.")
    return img


import time

def stereo_realtime_simulasi(img_l, img_r, n_frames=10):
    """Simulasi stereo matching real-time (parameter tuning)."""
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if len(img_l.shape)==3 else img_l
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if len(img_r.shape)==3 else img_r
    
    configs = [
        ("BM Fast", cv2.StereoBM_create(32, 9)),
        ("BM Quality", cv2.StereoBM_create(128, 21)),
        ("SGBM", cv2.StereoSGBM_create(0, 64, 5, 200, 800)),
    ]
    results = []
    for name, stereo in configs:
        times = []
        for _ in range(n_frames):
            t0 = time.time()
            disp = stereo.compute(gray_l, gray_r)
            times.append(time.time() - t0)
        avg_time = np.mean(times)
        fps = 1.0 / max(avg_time, 1e-6)
        results.append((name, disp.astype(np.float32)/16.0, avg_time, fps))
    return results


def main():
    """Fungsi utama: stereo matching realtime."""
    print("=" * 60)
    print("PERCOBAAN 16: STEREO MATCHING REALTIME")
    print("=" * 60)
    
    img_l = load_gambar("stereo_left.png")
    img_r = load_gambar("stereo_right.png")
    if img_l is None:
        img_l = np.random.randint(80,200,(300,400,3), dtype=np.uint8)
        for _ in range(15): cv2.circle(img_l,(np.random.randint(50,350),np.random.randint(50,250)),np.random.randint(5,25),(np.random.randint(0,255),)*3,-1)
        img_r = np.roll(img_l, -12, axis=1)
    
    results = stereo_realtime_simulasi(img_l, img_r)
    
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    for ax, (name, disp, t, fps) in zip(axes, results):
        ax.imshow(disp, cmap='jet'); ax.set_title(f"{name}\n{fps:.0f} FPS ({t*1000:.0f}ms)"); ax.axis('off')
        print(f"  {name}: {fps:.0f} FPS ({t*1000:.1f} ms)")
    plt.suptitle("Stereo Matching: Speed vs Quality", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_realtime.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/16_realtime.png")


if __name__ == "__main__":
    main()
