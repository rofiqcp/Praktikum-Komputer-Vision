# PROJECT MODUL 8: IMAGE STITCHING DAN ALIGNMENT

---

## Deskripsi Umum
Project mengintegrasikan seluruh konsep image stitching: feature matching, homography, warping, blending, exposure compensation, seam finding, dan projection. Pilih minimal 1 soal cerita.

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **360° Panorama Creator** — Buat panorama 360° penuh dari 12+ gambar.
2. **Automatic Panorama Ordering** — Otomatis tentukan urutan gambar dari kumpulan acak.
3. **HDR Panorama** — Gabungkan HDR imaging + stitching.
4. **Video Panorama** — Extract frames dari video → otomatis stitch ke panorama.
5. **Drone Image Mosaic** — Stitch gambar aerial (top-down).
6. **Document Scanner Panorama** — Stitch scan dokumen panjang (receipt, whiteboard).
7. **Multi-row Panorama** — Panorama 2D (horizontal + vertikal).
8. **Gigapixel Mosaic** — High-resolution panorama dari banyak gambar zoom-in.
9. **Moving Object Removal** — Hapus objek bergerak di area overlap.
10. **Custom Blending Mask** — User-defined mask untuk artistic panorama.
11. **Panorama with GPS** — Tagging panorama sections berdasarkan orientation.
12. **Quality Assessment Tool** — Hitung metrics kualitas stitching (PSNR, SSIM di overlap).
13. **Interactive Stitch Editor** — GUI untuk manual alignment correction.
14. **Exposure Bracketing Panorama** — Setiap posisi 3 exposure → HDR + stitch.
15. **Cylindrical Video Loop** — 360° cylindrical panorama dari rotating video.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Virtual Tour Generator
Agen properti ingin membuat virtual tour dari foto rumah. Buatlah: (a) ambil 8+ foto per ruangan (overlap 30%), (b) stitch panorama otomatis per ruangan, (c) cylindrical/spherical projection, (d) exposure compensation, (e) multi-band blending, (f) crop dan enhance, (g) output: panorama per ruangan + navigation.

### Soal 2: Sistem Dokumentasi Whiteboard
Kampus ingin sistem untuk mendokumentasikan catatan whiteboard panjang. Buatlah: (a) capture whiteboard dari beberapa foto, (b) perspective correction per foto, (c) stitch menjadi satu gambar panjang, (d) enhance (increase contrast, binarize), (e) OCR integrasi (opsional), (f) save sebagai PDF, (g) batch mode.

### Soal 3: Drone Mapping Application
Startup pertanian ingin memetakan lahan dari foto drone. Buatlah: (a) stitch gambar aerial (minimal 6 gambar top-down), (b) handle rotasi dan scale differences, (c) homography estimation, (d) multi-band blending, (e) overlay grid koordinat, (f) hitung area total, (g) export high-res mosaic.

### Soal 4: Panoramic Security Camera
Perusahaan keamanan ingin menggabungkan feed dari 3 kamera menjadi panoramic view. Buatlah: (a) stitch 3 gambar (simulasi multi-kamera), (b) real-time stitching (gunakan homography caching), (c) exposure compensation, (d) seamless blending, (e) overlay timestamp dan zona, (f) recording mode, (g) FPS display.

### Soal 5: Gigapixel Art Scanner
Museum ingin mendokumentasikan lukisan besar dalam resolusi tinggi. Buatlah: (a) ambil 9+ foto close-up lukisan (grid 3×3), (b) stitch menjadi satu gambar high-res, (c) multi-row stitching support, (d) color consistency (exposure compensation), (e) detail preservation (multi-band blending), (f) zoom viewer, (g) export berbagai resolusi.

### Soal 6: Sistem Pembuatan Peta Lantai
Arsitek ingin membuat peta lantai dari foto overhead. Buatlah: (a) ambil foto setiap bagian lantai, (b) stitch top-down, (c) perspective correction ke planar, (d) annotasi ruangan, (e) hitung area per ruangan, (f) scale calibration, (g) export DXF atau PDF.

### Soal 7: Street View Mini
Developer ingin membuat street view sederhana dari foto jalanan. Buatlah: (a) ambil 12+ foto 360° di satu lokasi, (b) spherical stitching, (c) equirectangular panorama, (d) viewer interaktif (pan/tilt dengan mouse), (e) multiple nodes (lokasi berbeda), (f) navigasi antar node, (g) thumbnail preview.

### Soal 8: Sistem Dokumentasi Scene Kecelakaan
Kepolisian ingin mendokumentasikan TKP secara menyeluruh. Buatlah: (a) panorama dari setiap sudut scene, (b) annotation (tanda, pengukuran), (c) multiple panorama per scene, (d) zooming dan detail shots stitching, (e) chronological ordering, (f) metadata (waktu, lokasi), (g) export report.

### Soal 9: Satellite Image Compositor
Lembaga cuaca ingin menggabungkan citra satelit. Buatlah: (a) download/simulasikan tile citra satelit, (b) stitch berdasarkan koordinat, (c) handle perbedaan waktu akuisisi (exposure), (d) cloud masking, (e) blend edges, (f) overlay informasi geografis, (g) temporal comparison.

### Soal 10: Photo Booth Panorama
Acara event ingin photo booth panoramic. Buatlah: (a) capture 3 foto (kiri, center, kanan), (b) real-time stitch, (c) exposure compensation, (d) border/frame decoration, (e) text overlay (nama event, tanggal), (f) save + print quality, (g) auto-capture mode (timer).

---

## Rubrik Penilaian Project

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Fungsionalitas | 35% | Semua fitur berjalan |
| Integrasi Percobaan | 20% | Menggunakan konsep ≥5 percobaan |
| Kualitas Kode | 15% | Clean, modular |
| Dokumentasi | 15% | README, screenshot |
| Kreativitas | 15% | Fitur tambahan, UI, optimasi |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project08.zip`
- **Isi**: Source code, dataset foto, README.md, contoh output panorama.
