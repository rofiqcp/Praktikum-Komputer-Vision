"""
Fix fallback synthetic image generation in all module experiments.
Replace synthetic generation fallbacks with proper exit() calls.
"""
import os
import re

BASE = r"D:\Praktikum-Komputer-Vision"

def read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [FIXED] {os.path.basename(path)}")


# ============================================================
# MODUL 05 - Fix experiments with synthetic fallback code
# ============================================================
DIR05 = os.path.join(BASE, "Modul05 - Deep Learning untuk Komputer Vision", "praktikum")

# --- Fix 10_deteksi_objek_sliding_window.py ---
path = os.path.join(DIR05, "10_deteksi_objek_sliding_window.py")
old = '''if img is None:
    # Mencoba memuat gambar alternatif
    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
    if img is None:
        # Membuat gambar sintetis dengan beberapa objek jika tidak ada gambar
        print("  [INFO] Membuat gambar sintetis dengan objek-objek...")
        img = np.ones((400, 600, 3), dtype=np.uint8) * 200

        # Menggambar beberapa persegi sebagai "objek"
        cv2.rectangle(img, (50, 50), (130, 130), (0, 0, 200), -1)
        cv2.rectangle(img, (200, 150), (280, 230), (0, 0, 180), -1)
        cv2.rectangle(img, (400, 80), (480, 160), (0, 0, 220), -1)
        cv2.rectangle(img, (300, 280), (380, 360), (0, 0, 190), -1)
        cv2.rectangle(img, (100, 300), (180, 380), (0, 0, 210), -1)

        # Menambahkan noise latar belakang untuk realisme
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)'''
new = '''if img is None:
    # Mencoba memuat gambar alternatif
    img = cv2.imread(os.path.join(IMAGE_DIR, "gedung.jpg"))
    if img is None:
        print("[ERROR] Gambar scene_outdoor.jpg dan gedung.jpg tidak ditemukan!")
        print("        Jalankan download_image.py terlebih dahulu.")
        exit()'''
content = read(path)
if old in content:
    write(path, content.replace(old, new))
else:
    print(f"  [WARN] Pattern not found in {os.path.basename(path)}")


# --- Fix 11_deteksi_objek_hog.py ---
path = os.path.join(DIR05, "11_deteksi_objek_hog.py")
content = read(path)
# Find and replace the synthetic fallback for pedestrian.jpg
old_pat = r'if img is None:\s*\n.*?# Membuat gambar sintetis.*?img = cv2\.add\(img, noise\)'
new_rep = '''if img is None:
        print("[ERROR] Gambar pedestrian.jpg tidak ditemukan!")
        print("        Jalankan download_image.py terlebih dahulu.")
        exit()'''
result = re.sub(old_pat, new_rep, content, flags=re.DOTALL)
if result != content:
    write(path, result)
else:
    # Try simpler approach - read and find the block
    lines = content.split('\n')
    start = None
    for i, line in enumerate(lines):
        if 'Membuat gambar sintetis' in line and 'pedestrian' in content[max(0, content.find(line)-500):content.find(line)]:
            start = i
            break
    if start:
        print(f"  [INFO] Found synthetic fallback in {os.path.basename(path)} but regex failed, trying manual")
    else:
        print(f"  [WARN] Pattern not found in {os.path.basename(path)}")


# --- Fix 12_yolo_konsep_grid.py ---
path = os.path.join(DIR05, "12_yolo_konsep_grid.py")
content = read(path)
# Find the synthetic fallback block after mobil.jpg fails
patterns_to_try = [
    ('img_scene', 'scene_traffic.jpg', 'scene_outdoor.jpg', 'mobil.jpg'),
]
# Search for the block
idx_start = content.find('if img_scene is None:')
if idx_start == -1:
    idx_start = content.find('if img is None:')
print(f"  [INFO] 12: img_scene is None at char {idx_start}")


# --- Fix 13_semantic_segmentation_manual.py ---
path = os.path.join(DIR05, "13_semantic_segmentation_manual.py")
content = read(path)
old = None
# Find the synthetic generation block
marker = 'np.zeros((400, 600, 3), dtype=np.uint8)'
idx = content.find(marker)
if idx > 0:
    # Find the if block that contains it
    # Look backward for "if"
    block_start = content.rfind('\nif ', 0, idx)
    block_end = content.find('\n# ==', idx)
    if block_start > 0 and block_end > 0:
        old_block = content[block_start:block_end]
        # Check it's the fallback (not something unrelated)
        if 'gedung' in old_block or 'sintetis' in old_block or 'Membuat' in old_block:
            new_block = '''
if img is None:
    print("[ERROR] Gambar scene_outdoor.jpg dan gedung.jpg tidak ditemukan!")
    print("        Jalankan download_image.py terlebih dahulu.")
    exit()'''
            write(path, content.replace(old_block, new_block))
        else:
            print(f"  [WARN] 13: Block found but not a fallback: {old_block[:100]}")
    else:
        print(f"  [WARN] 13: Could not find block boundaries")
else:
    print(f"  [WARN] 13: Marker not found in file")


# --- Fix 20_proyek_klasifikasi_bentuk.py ---
# This file uses 'segi_enam' but download script creates 'segi_enam' too - need to verify
path = os.path.join(DIR05, "20_proyek_klasifikasi_bentuk.py")
content = read(path)
# Find dataset categories
if "'segi_enam'" in content or '"segi_enam"' in content:
    print(f"  [OK] 20: file uses 'segi_enam' category - matches download script")
if "'elips'" in content or '"elips"' in content:
    print(f"  [WARN] 20: file uses 'elips' category - may need to update download script to match")
# Find the synthetic fallback for < 10 images
if 'sintetis' in content.lower() or 'synthetic' in content.lower() or 'Membuat' in content:
    print(f"  [INFO] 20: has synthetic fallback - check if it generates into dataset dir")


print("\nModul05 fixes attempted.")
print("="*60)

# ============================================================
# MODUL 06 - Fix experiments with synthetic fallback code
# ============================================================
DIR06 = os.path.join(BASE, "Modul06 - Recognition (Pengenalan)", "praktikum")

# --- Fix 19_recognition_pipeline_lengkap.py ---
path = os.path.join(DIR06, "19_recognition_pipeline_lengkap.py")
content = read(path)
old = '''    if test_image is None:
        # Membuat gambar wajah sederhana sebagai demonstrasi
        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 200
        cv2.ellipse(test_image, (200, 150), (60, 80), 0, 0, 360, (220, 190, 170), -1)
        cv2.circle(test_image, (175, 130), 8, (100, 80, 60), -1)
        cv2.circle(test_image, (225, 130), 8, (100, 80, 60), -1)
        cv2.ellipse(test_image, (200, 170), (20, 8), 0, 0, 180, (150, 80, 80), -1)'''
new = '''    if test_image is None:
        print("[ERROR] wajah_single.jpg tidak ditemukan!")
        print("        Jalankan download_image.py terlebih dahulu.")
        exit()'''
if old in content:
    write(path, content.replace(old, new))
else:
    # Try a broader search
    pattern = r'if test_image is None:\s*\n\s*#.*\n\s*test_image = np\.ones\(.*\)\s*\n.*cv2\.ellipse.*\n.*cv2\.circle.*\n.*cv2\.circle.*\n.*cv2\.ellipse'
    result = re.sub(pattern,
                    '    if test_image is None:\n        print("[ERROR] wajah_single.jpg tidak ditemukan!")\n        print("        Jalankan download_image.py terlebih dahulu.")\n        exit()',
                    content, flags=re.DOTALL)
    if result != content:
        write(path, result)
    else:
        print(f"  [WARN] 19/Modul06: Pattern not found - checking content...")
        idx = content.find('test_image = np.ones')
        if idx > 0:
            print(f"    Found np.ones at char {idx}: {content[idx:idx+80]}")


# --- Fix 20_proyek_recognition_sistem.py ---
path = os.path.join(DIR06, "20_proyek_recognition_sistem.py")
content = read(path)
changes = 0

# Fix 1: Face test fallback (wajah_single etc)
old1 = '''        if test_img is None:
            # Membuat gambar wajah sederhana
            test_img = np.ones((200, 200, 3), dtype=np.uint8) * 180
            cv2.ellipse(test_img, (100, 100), (50, 65), 0, 0, 360, (220, 190, 170), -1)
            cv2.circle(test_img, (80, 85), 6, (100, 80, 60), -1)
            cv2.circle(test_img, (120, 85), 6, (100, 80, 60), -1)'''
new1 = '''        if test_img is None:
            print(f"[WARNING] {test_file} tidak ditemukan, dilewati.")
            continue'''
# Try to find it
idx1 = content.find('test_img = np.ones((200, 200, 3)')
if idx1 > 0:
    # Find the if block that contains it
    blk_start = content.rfind('\n        if test_img is None:', 0, idx1)
    if blk_start > 0:
        blk_end = idx1 + 200
        # Find the end of this synthetic block
        next_section = content.find('\n\n', blk_end)
        old_blk = content[blk_start:next_section if next_section > 0 else blk_end]
        # Replace only synthesizing lines
        # Simple: replace if block up until end of draw calls
        print(f"  [INFO] 20/Modul06: Found face fallback block at char {blk_start}")
        changes += 1

# Fix 2: Gesture fallback (tangan_buka etc)
idx2 = content.find('tangan_buka')
if idx2 == -1:
    idx2 = content.find('tangan_open')
print(f"  [INFO] 20/Modul06: gesture file reference at char {idx2}")

# For Modul06/20, the fix needs to change filenames too:
# wajah_group.jpg -> wajah_grup.jpg (typo in experiments)
if 'wajah_group.jpg' in content:
    content = content.replace('wajah_group.jpg', 'wajah_grup.jpg')
    print(f"  [FIXED] 20/Modul06: wajah_group.jpg -> wajah_grup.jpg")

# teks_dokumen.jpg -> teks_printed.jpg
if 'teks_dokumen.jpg' in content:
    content = content.replace('teks_dokumen.jpg', 'teks_printed.jpg')
    print(f"  [FIXED] 20/Modul06: teks_dokumen.jpg -> teks_printed.jpg")

# teks_plat.jpg -> teks_scene.jpg
if 'teks_plat.jpg' in content:
    content = content.replace('teks_plat.jpg', 'teks_scene.jpg')
    print(f"  [FIXED] 20/Modul06: teks_plat.jpg -> teks_scene.jpg")

# scene_outdoor.jpg already exists in M06 download script

# tangan filenames
if 'tangan_buka.jpg' in content:
    content = content.replace('tangan_buka.jpg', 'tangan_open.jpg')
    print(f"  [FIXED] 20/Modul06: tangan_buka.jpg -> tangan_open.jpg")
if 'tangan_tutup.jpg' in content:
    content = content.replace('tangan_tutup.jpg', 'tangan_fist.jpg')
    print(f"  [FIXED] 20/Modul06: tangan_tutup.jpg -> tangan_fist.jpg")
if 'tangan_peace.jpg' in content:
    print(f"  [OK] 20/Modul06: tangan_peace.jpg - name matches")

write(path, content)


# --- Fix Modul06/11_hand_gesture_recognition.py ---
# It uses open.jpg, closed.jpg, pointing.jpg - need to map to our names
path = os.path.join(DIR06, "11_hand_gesture_recognition.py")
content = read(path)
changed = False
# Map filenames
for old_fn, new_fn in [('open.jpg', 'tangan_open.jpg'),
                        ('closed.jpg', 'tangan_fist.jpg'),
                        ('pointing.jpg', 'tangan_pointing.jpg')]:
    if f'"{old_fn}"' in content:
        content = content.replace(f'"{old_fn}"', f'"{new_fn}"')
        print(f"  [FIXED] 11/Modul06: {old_fn} -> {new_fn}")
        changed = True
    if f"'{old_fn}'" in content:
        content = content.replace(f"'{old_fn}'", f"'{new_fn}'")
        print(f"  [FIXED] 11/Modul06: '{old_fn}' -> '{new_fn}'")
        changed = True
if changed:
    write(path, content)
else:
    print(f"  [INFO] 11/Modul06: no filename changes needed")


# --- Fix Modul06/18_multi_face_tracking.py ---
# This file is FULLY SYNTHETIC - needs real image loading
path = os.path.join(DIR06, "18_multi_face_tracking.py")
content = read(path)
if 'imread' not in content:
    print(f"  [INFO] 18/Modul06: FULLY SYNTHETIC - needs wajah images")
    # Add image loading at the start after IMAGE_DIR definition
    old_hdr = '# Menampilkan header percobaan'
    if old_hdr not in content:
        old_hdr = 'print("=" * 60)'
    # Insert image loading before the main loop
    print(f"  [INFO] 18/Modul06: Would need to refactor to use wajah images")


print("\nModul06 fixes attempted.")
print("="*60)

# ============================================================
# MODUL 07 - Fix experiment 16 synthetic scene
# ============================================================
DIR07 = os.path.join(BASE, "Modul07 - Deteksi Fitur dan Pencocokan", "praktikum")

# File 16 generates a synthetic scene but the AR marker itself is loaded from file
# This is acceptable because the scene is a *demo* visualization, not a replacement
# However, the np.ones canvas + cv2.rectangle border is basically a background
# We should replace it with a real photo background
path = os.path.join(DIR07, "16_ar_marker_detection.py")
content = read(path)
markers = ['ar_marker.jpg']
for m in markers:
    if m in content:
        print(f"  [OK] 16/Modul07: uses {m} - real image reference correct")

print("\nModul07 check done.")
print("="*60)

print("\nAll fixes attempted!")
print("Summary: check output above for [FIXED], [WARN], [OK] status.")
