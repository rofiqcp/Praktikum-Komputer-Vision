"""
==========================================================================
PERCOBAAN 18: MULTI FACE TRACKING
==========================================================================
Tracking beberapa wajah pada video menggunakan deteksi + tracker sederhana.

Referensi: OpenCV docs
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


def deteksi_wajah(img):
    """Deteksi wajah menggunakan Haar Cascade."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))


def tracking_sederhana(prev_faces, curr_faces, max_dist=80):
    """
    Tracking wajah sederhana: matching berdasarkan jarak center.
    Assign ID berdasarkan kedekatan posisi antara frame.
    """
    assignments = {}
    used = set()
    for i, (px, py, pw, ph) in enumerate(prev_faces):
        pc = (px + pw//2, py + ph//2)
        best_j, best_d = -1, max_dist
        for j, (cx, cy, cw, ch) in enumerate(curr_faces):
            if j in used: continue
            cc = (cx + cw//2, cy + ch//2)
            d = np.sqrt((pc[0]-cc[0])**2 + (pc[1]-cc[1])**2)
            if d < best_d:
                best_d = d; best_j = j
        if best_j >= 0:
            assignments[best_j] = i
            used.add(best_j)
    return assignments


def main():
    """Fungsi utama: multi face tracking."""
    print("=" * 60)
    print("PERCOBAAN 18: MULTI FACE TRACKING")
    print("=" * 60)
    
    # Simulasi 5 frame dengan wajah bergerak
    np.random.seed(42)
    frames = []
    face_positions = [
        [(50,80,60,60), (200,100,60,60)],
        [(55,82,60,60), (195,98,60,60)],
        [(62,85,60,60), (188,95,60,60)],
        [(70,87,60,60), (180,93,60,60)],
        [(80,90,60,60), (170,90,60,60)],
    ]
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
    
    tracked_frames = []
    prev_faces = []
    face_ids = {}
    next_id = 0
    
    for t, positions in enumerate(face_positions):
        frame = np.ones((250, 350, 3), dtype=np.uint8) * 200
        for (x,y,w,h) in positions:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (150,130,120), -1)
            cv2.circle(frame, (x+w//2, y+h//3), 5, (60,50,40), -1)
        
        if prev_faces:
            assignments = tracking_sederhana(prev_faces, positions)
            new_ids = {}
            for j in range(len(positions)):
                if j in assignments:
                    new_ids[j] = face_ids.get(assignments[j], next_id)
                else:
                    new_ids[j] = next_id; next_id += 1
            face_ids = new_ids
        else:
            face_ids = {j: j for j in range(len(positions))}
            next_id = len(positions)
        
        display = frame.copy()
        for j, (x,y,w,h) in enumerate(positions):
            fid = face_ids.get(j, 0)
            cv2.rectangle(display, (x,y), (x+w,y+h), colors[fid % len(colors)], 2)
            cv2.putText(display, f"ID:{fid}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[fid%len(colors)], 1)
        
        cv2.putText(display, f"Frame {t+1}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        tracked_frames.append(display)
        prev_faces = positions
    
    fig, axes = plt.subplots(1, len(tracked_frames), figsize=(15, 4))
    for ax, frame in zip(axes, tracked_frames):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); ax.axis('off')
    plt.suptitle("Multi Face Tracking (ID Assignment)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_multi_tracking.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/18_multi_tracking.png")
    print("\nRINGKASAN: Face tracking = deteksi + assignment ID berdasarkan jarak antar frame.")


if __name__ == "__main__":
    main()
