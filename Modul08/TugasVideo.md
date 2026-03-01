# TUGAS VIDEO MODUL 8: IMAGE STITCHING DAN ALIGNMENT

---

## Deskripsi Tugas
Buat video laporan yang mendemonstrasikan seluruh materi, 10 percobaan, dan project image stitching. Tunjukkan proses end-to-end dari pengambilan foto hingga panorama final.

---

## Struktur Video

### 1. Pembukaan (Maks 2 menit)
- Perkenalan: Nama, NIM, kelas, modul.
- Overview topik Image Stitching.

### 2. Penjelasan Materi (10–15 menit)
- Pipeline stitching: detection → matching → homography → warp → blend.
- Motion models: translation, affine, homography.
- Projection: planar vs cylindrical vs spherical.
- Blending: feather, multi-band (Laplacian pyramid).
- Bundle adjustment, exposure compensation, seam finding.
- **Wajib**: Diagram pipeline + perbandingan proyeksi.

### 3. Demo 10 Percobaan (30–40 menit)

| No | Percobaan | Poin Penting |
|----|-----------|--------------|
| 1 | Manual Stitching Pipeline | Feature match → homography → warp |
| 2 | OpenCV Stitcher API | Automatic vs manual comparison |
| 3 | Blending Comparison | No blend vs feather vs multi-band |
| 4 | Multi-Image Panorama | 5+ gambar, referensi tengah |
| 5 | Cylindrical Projection | Planar vs cylindrical comparison |
| 6 | Spherical Projection | 3 projections side-by-side |
| 7 | Bundle Adjustment | Before/after BA comparison |
| 8 | Exposure Compensation | Different exposure → compensated |
| 9 | Seam Finding | Voronoi vs GraphCut visual |
| 10 | Real-time/Interactive Stitching | Live demo, FPS display |

- **PENTING**: Tunjukkan pengambilan foto asli (video diri sendiri mengambil foto outdoor/indoor).

### 4. Demo Project (10–15 menit)
- Demo project soal cerita.
- Tunjukkan input → proses → output panorama.

### 5. Analisis dan Penutup (5 menit)
- Rangkuman perbandingan metode.
- Tantangan dan solusi.
- Kesimpulan.

---

## Ketentuan Teknis

| Aspek | Ketentuan |
|-------|-----------|
| Durasi | 60–75 menit |
| Resolusi | Minimal 1080p |
| Recording | Screen recording + webcam |
| Webcam | Tunjukkan diri saat menjelaskan + saat mengambil foto |
| Audio | Narasi jelas |
| Platform | YouTube (Unlisted) atau Google Drive |
| Format | MP4 |

---

## Rubrik Penilaian Video

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Pembukaan | 5% | Profesional |
| Penjelasan Materi | 15% | Akurat, diagram jelas |
| Demo 10 Percobaan | 40% | Semua berjalan |
| Demo Project | 20% | Fitur lengkap |
| Analisis & Kesimpulan | 10% | Kritis, kuantitatif |
| Kualitas Video | 10% | Resolusi, audio |

### Bonus & Penalti
| Item | Nilai |
|------|-------|
| Tunjukkan proses pengambilan foto sendiri | +5 |
| Demo 360° panorama | +5 |
| Video < 45 menit | -10 |
| Tidak ada webcam | -5 |
| Audio tidak jelas | -5 |
| Percobaan error tanpa penjelasan | -5 per percobaan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: Link YouTube (Unlisted) atau Google Drive.
- **Naming**: `[NIM]_[Nama]_Video_Modul08`
