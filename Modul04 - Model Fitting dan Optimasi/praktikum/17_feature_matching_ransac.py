"""
==========================================================================
PERCOBAAN 17: FEATURE MATCHING DAN RANSAC PIPELINE
==========================================================================
Menggabungkan deteksi fitur, deskriptor matching, dan estimasi homografi
menggunakan RANSAC. Ini adalah pipeline praktis yang banyak digunakan
dalam image stitching, object recognition, dan visual SLAM.

Fungsi utama:
- cv2.ORB_create()         : detektor + deskriptor ORB (cepat, gratis)
- cv2.BFMatcher()          : brute-force matching
- cv2.drawMatches()        : visualisasi pasangan fitur
- cv2.findHomography()     : estimasi homografi + RANSAC
- cv2.warpPerspective()    : warping perspektif
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

print("=" * 60)
print("PERCOBAAN 17: FEATURE MATCHING DAN RANSAC PIPELINE")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Memuat dua gambar (scene A dan scene B)
# ============================================================
print("\n--- 1. Memuat Gambar ---")

img_a_path = os.path.join(IMAGE_DIR, "frame1.png")
img_b_path = os.path.join(IMAGE_DIR, "frame2.png")

if not os.path.exists(img_a_path) or not os.path.exists(img_b_path):
    print("[ERROR] frame1.png / frame2.png tidak ditemukan. Jalankan download_image.py!"); exit()

img_a = cv2.imread(img_a_path)
img_b = cv2.imread(img_b_path)
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

print(f"  Scene A: {img_a.shape}")
print(f"  Scene B: {img_b.shape}")

# ============================================================
# 2. Deteksi Fitur dengan ORB
# ============================================================
print("\n--- 2. Deteksi Fitur ORB ---")

# cv2.ORB_create: Oriented FAST and Rotated BRIEF
orb = cv2.ORB_create(
    nfeatures=500,        # maksimum fitur
    scaleFactor=1.2,      # faktor skala antar level piramida
    nlevels=8,            # jumlah level piramida
    edgeThreshold=31,     # threshold tepi
    patchSize=31          # ukuran patch untuk deskriptor
)

# detectAndCompute: deteksi keypoint + hitung deskriptor
kp_a, des_a = orb.detectAndCompute(gray_a, None)
kp_b, des_b = orb.detectAndCompute(gray_b, None)

print(f"  Scene A: {len(kp_a)} keypoints")
print(f"  Scene B: {len(kp_b)} keypoints")

# Gambar keypoints
img_kp_a = cv2.drawKeypoints(img_a, kp_a, None, color=(0, 255, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_kp_b = cv2.drawKeypoints(img_b, kp_b, None, color=(0, 255, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite(os.path.join(OUTPUT_DIR, "17_keypoints_a.png"), img_kp_a)
cv2.imwrite(os.path.join(OUTPUT_DIR, "17_keypoints_b.png"), img_kp_b)

# ============================================================
# 3. Feature Matching dengan BFMatcher
# ============================================================
print("\n--- 3. Feature Matching ---")

if des_a is not None and des_b is not None:
    # cv2.BFMatcher: brute-force matcher
    # NORM_HAMMING untuk deskriptor biner (ORB, BRIEF)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # knnMatch: cari k=2 match terdekat untuk setiap deskriptor
    matches_knn = bf.knnMatch(des_a, des_b, k=2)
    
    print(f"  Total raw matches: {len(matches_knn)}")
    
    # ============================================================
    # 4. Ratio Test (Lowe's ratio test)
    # ============================================================
    print("\n--- 4. Lowe's Ratio Test ---")
    
    # Filter match menggunakan ratio test
    # Jika match terbaik jauh lebih baik dari match kedua → match valid
    ratio_threshold = 0.75
    good_matches = []
    
    for m, n in matches_knn:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    print(f"  Good matches (ratio < {ratio_threshold}): {len(good_matches)}")
    
    # Gambar good matches
    img_matches = cv2.drawMatches(img_a, kp_a, img_b, kp_b, good_matches, None,
                                   matchColor=(0, 255, 0),
                                   singlePointColor=(255, 0, 0),
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "17_good_matches.png"), img_matches)
    
    # ============================================================
    # 5. Estimasi Homography dengan RANSAC
    # ============================================================
    print("\n--- 5. Homography Estimation ---")
    
    if len(good_matches) >= 4:
        # Ambil koordinat titik dari match
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # cv2.findHomography dengan RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        
        inlier_count = np.sum(mask)
        total = len(mask)
        ratio = inlier_count / total * 100
        
        print(f"  Inliers: {inlier_count}/{total} ({ratio:.1f}%)")
        print(f"  Homography matrix:")
        if H is not None:
            for row in H:
                print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
        
        # Gambar inlier matches saja
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matches_mask,
            flags=2
        )
        img_inliers = cv2.drawMatches(img_a, kp_a, img_b, kp_b,
                                       good_matches, None, **draw_params)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "17_inlier_matches.png"), img_inliers)
        
        # ============================================================
        # 6. Warping menggunakan Homography
        # ============================================================
        print("\n--- 6. Warping ---")
        
        h_img, w_img = img_a.shape[:2]
        
        # Warp scene A ke perspective scene B
        warped_a = cv2.warpPerspective(img_a, H, (w_img, h_img))
        cv2.imwrite(os.path.join(OUTPUT_DIR, "17_warped_a.png"), warped_a)
        
        # Overlay: blend warped_a dan img_b
        overlay = cv2.addWeighted(warped_a, 0.5, img_b, 0.5, 0)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "17_overlay.png"), overlay)
        
        print(f"  Warped dan overlay disimpan")
        
        # ============================================================
        # 7. Perbandingan RANSAC vs LMEDS
        # ============================================================
        print("\n--- 7. Perbandingan Metode ---")
        
        methods = {
            'RANSAC': cv2.RANSAC,
            'LMEDS': cv2.LMEDS,
            'RHO': cv2.RHO,
        }
        
        for name, method in methods.items():
            try:
                H_m, mask_m = cv2.findHomography(src_pts, dst_pts, method, 5.0)
                if mask_m is not None:
                    inliers = np.sum(mask_m)
                    print(f"  {name}: inliers={inliers}/{len(mask_m)} "
                          f"({inliers/len(mask_m)*100:.1f}%)")
                else:
                    print(f"  {name}: gagal")
            except Exception as e:
                print(f"  {name}: error - {e}")
    else:
        print(f"  Tidak cukup match ({len(good_matches)} < 4)")
        H = None
else:
    print("  Deskriptor tidak ditemukan!")
    good_matches = []
    H = None

# ============================================================
# 8. Visualisasi gabungan
# ============================================================
print("\n--- 8. Visualisasi Gabungan ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(cv2.cvtColor(img_kp_a, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"Scene A ({len(kp_a)} keypoints)")
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img_kp_b, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"Scene B ({len(kp_b)} keypoints)")
axes[0, 1].axis('off')

if len(good_matches) > 0:
    axes[0, 2].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"Good Matches ({len(good_matches)})")
else:
    axes[0, 2].text(0.5, 0.5, "No matches", ha='center', va='center')
axes[0, 2].axis('off')

if H is not None:
    axes[1, 0].imshow(cv2.cvtColor(img_inliers, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Inliers ({inlier_count}/{total})")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(warped_a, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Warped A → B")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Overlay (blended)")
    axes[1, 2].axis('off')
else:
    for ax in axes[1, :]:
        ax.text(0.5, 0.5, "Homography gagal", ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "17_feature_matching_all.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Disimpan: {output_path}")

print("\n" + "=" * 60)
print("PERCOBAAN 17 SELESAI")
print("=" * 60)
