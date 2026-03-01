<div align="center">

# 🎓 Praktikum Computer Vision

**Kurikulum praktikum lengkap berbasis buku teks**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Modules](https://img.shields.io/badge/Modul-12-blue?style=flat-square)]()
[![Programs](https://img.shields.io/badge/Program_Python-240%2B-orange?style=flat-square)]()

> Berdasarkan buku **"Computer Vision: Algorithms and Applications, 2nd Edition"** oleh **Richard Szeliski**  
> 12 modul | 240+ program Python | Dari dasar hingga 3D reconstruction & neural rendering

</div>

---

## 📑 Daftar Isi

- [Highlight](#-highlight)
- [Daftar Modul](#-daftar-modul)
- [Detail Setiap Modul](#-detail-setiap-modul)
- [Instalasi & Setup](#-instalasi--setup)
- [Cara Belajar](#-cara-belajar-yang-disarankan)
- [Struktur Repo](#-struktur-repo)
- [Referensi](#-referensi)
- [Kontribusi](#-kontribusi)

---

## ✨ Highlight

| Fitur | Keterangan |
|-------|-----------|
| 📦 **12 modul lengkap** | Dari dasar hingga 3D reconstruction & image-based rendering |
| 🐍 **240+ program Python** | Setiap modul berisi 20 file praktikum terstruktur |
| 📘 **Materi + Jobsheet + Project** | Setiap modul dilengkapi teori, panduan, dan proyek akhir |
| 🎥 **Tugas Video** | Setiap modul punya panduan tugas video presentasi |
| 📚 **Referensi PDF lengkap** | Buku utama Szeliski + 15 referensi tambahan |
| 🔬 **Review per modul** | Analisis kelengkapan & saran pengembangan |
| ⚙️ **Satu paket dependensi** | Semua library via `requirements.txt` |

---

## 📚 Daftar Modul

| # | Modul | Topik | Program |
|---|-------|-------|---------|
| 01 | [Pendahuluan CV](Modul01%20-%20Pendahuluan%20Komputer%20Vision/) | Konsep dasar, pipeline, pengantar tools | [20 program](Modul01%20-%20Pendahuluan%20Komputer%20Vision/praktikum/) |
| 02 | [Pembentukan Citra](Modul02%20-%20Pembentukan%20Citra%20(Image%20Formation)/) | Kamera, transformasi geometrik, kalibrasi | [20 program](Modul02%20-%20Pembentukan%20Citra%20(Image%20Formation)/praktikum/) |
| 03 | [Pemrosesan Citra](Modul03%20-%20Pemrosesan%20Citra%20(Image%20Processing)/) | Filtering, morfologi, Fourier, thresholding | [20 program](Modul03%20-%20Pemrosesan%20Citra%20(Image%20Processing)/praktikum/) |
| 04 | [Model Fitting & Optimasi](Modul04%20-%20Model%20Fitting%20dan%20Optimasi/) | RANSAC, Hough, homography, optimisasi | [20 program](Modul04%20-%20Model%20Fitting%20dan%20Optimasi/praktikum/) |
| 05 | [Deep Learning CV](Modul05%20-%20Deep%20Learning%20untuk%20Komputer%20Vision/) | CNN, YOLO, transfer learning, segmentasi | [20 program](Modul05%20-%20Deep%20Learning%20untuk%20Komputer%20Vision/praktikum/) |
| 06 | [Recognition](Modul06%20-%20Recognition%20(Pengenalan)/) | Wajah, OCR, pedestrian, evaluasi metrik | [20 program](Modul06%20-%20Recognition%20(Pengenalan)/praktikum/) |
| 07 | [Deteksi Fitur & Pencocokan](Modul07%20-%20Deteksi%20Fitur%20dan%20Pencocokan/) | SIFT, ORB, AKAZE, FLANN, AR marker | [20 program](Modul07%20-%20Deteksi%20Fitur%20dan%20Pencocokan/praktikum/) |
| 08 | [Image Stitching](Modul08%20-%20Image%20Stitching%20dan%20Alignment/) | Panorama, blending, cylindrical, seam | [20 program](Modul08%20-%20Image%20Stitching%20dan%20Alignment/praktikum/) |
| 09 | [Estimasi Gerak](Modul09%20-%20Estimasi%20Gerak%20(Motion%20Estimation)/) | Optical flow, tracking, video stabilization | [20 program](Modul09%20-%20Estimasi%20Gerak%20(Motion%20Estimation)/praktikum/) |
| 10 | [Computational Photography](Modul10%20-%20Computational%20Photography/) | HDR, denoising, inpainting, style transfer | [20 program](Modul10%20-%20Computational%20Photography/praktikum/) |
| 11 | [SfM & Depth Estimation](Modul11%20-%20Structure%20from%20Motion%20dan%20Depth%20Estimation/) | F/E matrix, stereo, disparity, point cloud | [20 program](Modul11%20-%20Structure%20from%20Motion%20dan%20Depth%20Estimation/praktikum/) |
| 12 | [Rekonstruksi 3D & IBR](Modul12%20-%20Rekonstruksi%203D%20dan%20Image-Based%20Rendering/) | ICP, mesh, NeRF intro, view synthesis | [20 program](Modul12%20-%20Rekonstruksi%203D%20dan%20Image-Based%20Rendering/praktikum/) |

---

## 🔍 Detail Setiap Modul

<details>
<summary><strong>Modul 01 — Pendahuluan Komputer Vision</strong></summary>

> **Referensi:** Szeliski Ch.1 | **Dokumen:** [Materi](Modul01%20-%20Pendahuluan%20Komputer%20Vision/Materi.md) · [Jobsheet](Modul01%20-%20Pendahuluan%20Komputer%20Vision/Jobsheet.md) · [Project](Modul01%20-%20Pendahuluan%20Komputer%20Vision/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_loading_dan_menampilkan_gambar.py` | Load & tampilkan gambar |
| 02 | `02_properti_gambar.py` | Shape, dtype, dimensi |
| 03 | `03_konversi_ruang_warna.py` | BGR, RGB, HSV, Gray, Lab |
| 04 | `04_akses_manipulasi_piksel.py` | Akses & edit piksel |
| 05 | `05_operasi_aritmatika_gambar.py` | Add, subtract, multiply |
| 06 | `06_operasi_bitwise.py` | AND, OR, XOR, NOT |
| 07 | `07_menggambar_bentuk_geometris.py` | Garis, lingkaran, poligon |
| 08 | `08_menulis_teks_pada_gambar.py` | Anotasi teks |
| 09 | `09_region_of_interest.py` | ROI crop & mask |
| 10 | `10_resize_dan_scaling.py` | Resize dengan interpolasi |
| 11 | `11_cropping_gambar.py` | Cropping region |
| 12 | `12_rotasi_gambar.py` | Rotasi berbagai sudut |
| 13 | `13_flip_gambar.py` | Flip horizontal/vertikal |
| 14 | `14_padding_border_gambar.py` | Border & padding |
| 15 | `15_splitting_merging_channel.py` | Split & merge B/G/R |
| 16 | `16_blending_dua_gambar.py` | Alpha blending |
| 17 | `17_brightness_dan_contrast.py` | Kontrol kecerahan |
| 18 | `18_histogram_gambar.py` | Histogram plot |
| 19 | `19_masking_gambar.py` | Masking warna |
| 20 | `20_menyimpan_berbagai_format.py` | Simpan JPG/PNG/BMP/TIFF |

</details>

<details>
<summary><strong>Modul 02 — Pembentukan Citra (Image Formation)</strong></summary>

> **Referensi:** Szeliski Ch.2 | **Dokumen:** [Materi](Modul02%20-%20Pembentukan%20Citra%20(Image%20Formation)/Materi.md) · [Jobsheet](Modul02%20-%20Pembentukan%20Citra%20(Image%20Formation)/Jobsheet.md) · [Project](Modul02%20-%20Pembentukan%20Citra%20(Image%20Formation)/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_translasi_gambar.py` | Translasi 2D |
| 02 | `02_rotasi_sudut_bebas.py` | Rotasi bebas dengan pusat pivot |
| 03 | `03_scaling_zoom.py` | Scaling & zoom |
| 04 | `04_transformasi_affine.py` | Transformasi affine |
| 05 | `05_transformasi_perspektif.py` | Homography perspektif |
| 06 | `06_kalibrasi_kamera_checkerboard.py` | Kalibrasi kamera |
| 07 | `07_proyeksi_3d_ke_2d.py` | Proyeksi 3D ke bidang 2D |
| 08 | `08_koreksi_distorsi_lensa.py` | Koreksi distorsi radial |
| 09 | `09_sampling_dan_aliasing.py` | Aliasing & Nyquist |
| 10 | `10_gamma_correction.py` | Koreksi gamma |
| 11 | `11_interpolasi_gambar.py` | Nearest, bilinear, bicubic |
| 12 | `12_konversi_koordinat_polar.py` | Polar ↔ Cartesian |
| 13 | `13_shearing_gambar.py` | Shear transform |
| 14 | `14_refleksi_gambar.py` | Refleksi bidang |
| 15 | `15_matriks_transformasi_homogen.py` | Homogeneous matrix |
| 16 | `16_image_pyramid.py` | Gaussian & Laplacian pyramid |
| 17 | `17_barrel_pincushion_distortion.py` | Distorsi barrel/pincushion |
| 18 | `18_remapping_gambar.py` | Remap piksel custom |
| 19 | `19_transformasi_log_dan_power.py` | Log & power transform |
| 20 | `20_pembuatan_citra_sintetis.py` | Generate citra sintetis |

</details>

<details>
<summary><strong>Modul 03 — Pemrosesan Citra (Image Processing)</strong></summary>

> **Referensi:** Szeliski Ch.3 | **Dokumen:** [Materi](Modul03%20-%20Pemrosesan%20Citra%20(Image%20Processing)/Materi.md) · [Jobsheet](Modul03%20-%20Pemrosesan%20Citra%20(Image%20Processing)/Jobsheet.md) · [Project](Modul03%20-%20Pemrosesan%20Citra%20(Image%20Processing)/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_brightness_dan_contrast.py` | Linear brightness/contrast |
| 02 | `02_histogram_equalization.py` | Equalisasi histogram |
| 03 | `03_clahe.py` | CLAHE adaptive equalization |
| 04 | `04_gamma_correction.py` | Koreksi gamma |
| 05 | `05_thresholding_global.py` | Binary, trunc, tozero |
| 06 | `06_thresholding_otsu_triangle.py` | Auto threshold Otsu & Triangle |
| 07 | `07_adaptive_thresholding.py` | Mean & Gaussian adaptive |
| 08 | `08_konvolusi_dan_filter2d.py` | Custom kernel konvolusi |
| 09 | `09_gaussian_blur.py` | Gaussian smoothing |
| 10 | `10_median_dan_bilateral_filter.py` | Median & bilateral |
| 11 | `11_sharpening.py` | Unsharp mask & Laplacian |
| 12 | `12_deteksi_tepi_sobel.py` | Sobel gradient |
| 13 | `13_deteksi_tepi_canny.py` | Canny edge |
| 14 | `14_deteksi_tepi_laplacian.py` | Laplacian of Gaussian |
| 15 | `15_morfologi_erosi_dilasi.py` | Erosi & dilasi |
| 16 | `16_morfologi_lanjut.py` | Opening, closing, gradient |
| 17 | `17_tophat_blackhat.py` | Top-hat & black-hat |
| 18 | `18_transformasi_fourier.py` | FFT & visualisasi frekuensi |
| 19 | `19_filter_frekuensi.py` | Low-pass & high-pass filter |
| 20 | `20_alpha_blending_compositing.py` | Alpha compositing |

</details>

<details>
<summary><strong>Modul 04 — Model Fitting dan Optimasi</strong></summary>

> **Referensi:** Szeliski Ch.4 | **Dokumen:** [Materi](Modul04%20-%20Model%20Fitting%20dan%20Optimasi/Materi.md) · [Jobsheet](Modul04%20-%20Model%20Fitting%20dan%20Optimasi/Jobsheet.md) · [Project](Modul04%20-%20Model%20Fitting%20dan%20Optimasi/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_ordinary_least_squares.py` | OLS fitting garis |
| 02 | `02_weighted_least_squares.py` | Weighted LS |
| 03 | `03_total_least_squares.py` | Total LS (orthogonal) |
| 04 | `04_ransac_fitting_garis.py` | RANSAC robust line |
| 05 | `05_ransac_fitting_lingkaran.py` | RANSAC circle |
| 06 | `06_hough_transform_garis.py` | Hough line transform |
| 07 | `07_hough_transform_lingkaran.py` | Hough circle |
| 08 | `08_homography_estimation.py` | Estimasi homography |
| 09 | `09_koreksi_perspektif_dokumen.py` | Document deskewing |
| 10 | `10_irls_robust_fitting.py` | IRLS iterative |
| 11 | `11_regularisasi_ridge_lasso.py` | Ridge & Lasso |
| 12 | `12_fitting_ellips_kontur.py` | Ellipse dari kontur |
| 13 | `13_template_matching.py` | Template matching |
| 14 | `14_graph_cut_segmentation.py` | GrabCut segmentation |
| 15 | `15_lucas_kanade_optical_flow.py` | LK optical flow |
| 16 | `16_dense_optical_flow.py` | Farneback dense flow |
| 17 | `17_feature_matching_ransac.py` | Feature + RANSAC pipeline |
| 18 | `18_cross_validation_model_selection.py` | CV & model selection |
| 19 | `19_denoising_optimasi.py` | Denoising via optimasi |
| 20 | `20_pipeline_gabungan.py` | Pipeline end-to-end |

</details>

<details>
<summary><strong>Modul 05 — Deep Learning untuk Komputer Vision</strong></summary>

> **Referensi:** Szeliski Ch.5 | **Dokumen:** [Materi](Modul05%20-%20Deep%20Learning%20untuk%20Komputer%20Vision/Materi.md) · [Jobsheet](Modul05%20-%20Deep%20Learning%20untuk%20Komputer%20Vision/Jobsheet.md) · [Project](Modul05%20-%20Deep%20Learning%20untuk%20Komputer%20Vision/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_opencv_dnn_blob.py` | Blob dari gambar (DNN input) |
| 02 | `02_opencv_dnn_klasifikasi.py` | Klasifikasi dengan OpenCV DNN |
| 03 | `03_perbandingan_model_pretrained.py` | Perbandingan model pre-trained |
| 04 | `04_arsitektur_cnn_visualisasi.py` | Visualisasi arsitektur CNN |
| 05 | `05_fungsi_aktivasi.py` | ReLU, Sigmoid, Tanh |
| 06 | `06_konvolusi_dan_pooling.py` | Conv & pooling manual |
| 07 | `07_data_augmentasi_dasar.py` | Augmentasi data |
| 08 | `08_transfer_learning_konsep.py` | Transfer learning pipeline |
| 09 | `09_backpropagation_visualisasi.py` | Visualisasi backprop |
| 10 | `10_deteksi_objek_sliding_window.py` | Sliding window detector |
| 11 | `11_deteksi_objek_hog.py` | HOG + SVM detector |
| 12 | `12_yolo_konsep_grid.py` | YOLO grid concept |
| 13 | `13_semantic_segmentation_manual.py` | Segmentasi pixel-wise |
| 14 | `14_instance_segmentation_konsep.py` | Instance segmentation |
| 15 | `15_loss_function_visualisasi.py` | Loss landscape |
| 16 | `16_optimizer_visualisasi.py` | SGD, Adam, RMSprop |
| 17 | `17_batch_normalization_dropout.py` | BN & Dropout |
| 18 | `18_model_evaluasi_metrik.py` | mAP, IoU, F1 |
| 19 | `19_onnx_dan_deployment_konsep.py` | ONNX export & inference |
| 20 | `20_proyek_klasifikasi_bentuk.py` | Proyek: klasifikasi bentuk |

</details>

<details>
<summary><strong>Modul 06 — Recognition (Pengenalan)</strong></summary>

> **Referensi:** Szeliski Ch.6 | **Dokumen:** [Materi](Modul06%20-%20Recognition%20(Pengenalan)/Materi.md) · [Jobsheet](Modul06%20-%20Recognition%20(Pengenalan)/Jobsheet.md) · [Project](Modul06%20-%20Recognition%20(Pengenalan)/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_face_detection_haar_cascade.py` | Deteksi wajah Haar |
| 02 | `02_face_detection_dnn_konsep.py` | Deteksi wajah DNN |
| 03 | `03_face_recognition_lbph.py` | Pengenalan wajah LBPH |
| 04 | `04_face_recognition_eigenfaces.py` | Eigenfaces PCA |
| 05 | `05_face_landmark_detection.py` | 68-point face landmark |
| 06 | `06_ocr_preprocessing.py` | Preprocessing untuk OCR |
| 07 | `07_ocr_tesseract_konsep.py` | Tesseract OCR |
| 08 | `08_scene_text_detection.py` | Deteksi teks di scene |
| 09 | `09_pedestrian_detection_hog.py` | HOG pedestrian |
| 10 | `10_vehicle_detection.py` | Deteksi kendaraan |
| 11 | `11_hand_gesture_recognition.py` | Gesture tangan |
| 12 | `12_object_classification_bovw.py` | Bag of Visual Words |
| 13 | `13_scene_recognition.py` | Pengenalan scene |
| 14 | `14_face_embedding_distance.py` | Face embedding & distance |
| 15 | `15_evaluasi_classification_metrics.py` | Precision, Recall, F1 |
| 16 | `16_evaluasi_detection_metrics.py` | mAP & IoU |
| 17 | `17_recognition_roc_curve.py` | ROC & AUC curve |
| 18 | `18_multi_face_tracking.py` | Multi-face tracking |
| 19 | `19_recognition_pipeline_lengkap.py` | Pipeline lengkap |
| 20 | `20_proyek_recognition_sistem.py` | Proyek: sistem recognition |

</details>

<details>
<summary><strong>Modul 07 — Deteksi Fitur dan Pencocokan</strong></summary>

> **Referensi:** Szeliski Ch.7 | **Dokumen:** [Materi](Modul07%20-%20Deteksi%20Fitur%20dan%20Pencocokan/Materi.md) · [Jobsheet](Modul07%20-%20Deteksi%20Fitur%20dan%20Pencocokan/Jobsheet.md) · [Project](Modul07%20-%20Deteksi%20Fitur%20dan%20Pencocokan/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_harris_corner_detection.py` | Harris corner |
| 02 | `02_shi_tomasi_corner.py` | Shi-Tomasi corner |
| 03 | `03_sift_feature_detection.py` | SIFT keypoint & descriptor |
| 04 | `04_orb_feature_detection.py` | ORB (binary descriptor) |
| 05 | `05_akaze_dan_fast.py` | AKAZE & FAST |
| 06 | `06_brute_force_matching.py` | Brute-force BFMatcher |
| 07 | `07_flann_matching_ratio_test.py` | FLANN + Lowe ratio test |
| 08 | `08_homography_ransac.py` | Homography via RANSAC |
| 09 | `09_object_detection_feature.py` | Deteksi objek berbasis fitur |
| 10 | `10_feature_invariance_rotasi.py` | Invariansi terhadap rotasi |
| 11 | `11_feature_invariance_skala.py` | Invariansi terhadap skala |
| 12 | `12_feature_invariance_iluminasi.py` | Invariansi iluminasi |
| 13 | `13_deskriptor_perbandingan.py` | Perbandingan deskriptor |
| 14 | `14_geometric_verification_detail.py` | Geometric verification |
| 15 | `15_image_retrieval.py` | Image retrieval by features |
| 16 | `16_ar_marker_detection.py` | AR marker detection |
| 17 | `17_keypoint_repeatability.py` | Repeatability evaluation |
| 18 | `18_multi_image_matching.py` | Matching multi-image |
| 19 | `19_feature_matching_pipeline.py` | Full pipeline matching |
| 20 | `20_proyek_feature_matching_app.py` | Proyek: feature matching app |

</details>

<details>
<summary><strong>Modul 08 — Image Stitching dan Alignment</strong></summary>

> **Referensi:** Szeliski Ch.8 | **Dokumen:** [Materi](Modul08%20-%20Image%20Stitching%20dan%20Alignment/Materi.md) · [Jobsheet](Modul08%20-%20Image%20Stitching%20dan%20Alignment/Jobsheet.md) · [Project](Modul08%20-%20Image%20Stitching%20dan%20Alignment/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_stitching_manual_pipeline.py` | Pipeline stitching manual |
| 02 | `02_opencv_stitcher_api.py` | OpenCV Stitcher API |
| 03 | `03_teknik_blending_perbandingan.py` | Perbandingan teknik blending |
| 04 | `04_multi_image_panorama.py` | Panorama multi-image |
| 05 | `05_cylindrical_projection.py` | Proyeksi silindris |
| 06 | `06_spherical_projection.py` | Proyeksi sferis |
| 07 | `07_bundle_adjustment_konsep.py` | Konsep bundle adjustment |
| 08 | `08_exposure_compensation.py` | Kompensasi exposure |
| 09 | `09_seam_finding.py` | Seam finding & cutting |
| 10 | `10_realtime_interactive_stitching.py` | Stitching real-time |
| 11 | `11_panorama_cropping_auto.py` | Auto-crop panorama |
| 12 | `12_homography_estimation_visualisasi.py` | Visualisasi homography |
| 13 | `13_image_registration.py` | Image registration |
| 14 | `14_laplacian_pyramid_blending_detail.py` | Laplacian pyramid blend |
| 15 | `15_gain_compensation_manual.py` | Gain compensation manual |
| 16 | `16_seam_quality_evaluation.py` | Evaluasi kualitas seam |
| 17 | `17_panorama_loop_closure.py` | Loop closure panorama |
| 18 | `18_document_stitching.py` | Stitching dokumen |
| 19 | `19_panorama_pipeline_lengkap.py` | Pipeline lengkap |
| 20 | `20_proyek_panorama_app.py` | Proyek: aplikasi panorama |

</details>

<details>
<summary><strong>Modul 09 — Estimasi Gerak (Motion Estimation)</strong></summary>

> **Referensi:** Szeliski Ch.9 | **Dokumen:** [Materi](Modul09%20-%20Estimasi%20Gerak%20(Motion%20Estimation)/Materi.md) · [Jobsheet](Modul09%20-%20Estimasi%20Gerak%20(Motion%20Estimation)/Jobsheet.md) · [Project](Modul09%20-%20Estimasi%20Gerak%20(Motion%20Estimation)/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_optical_flow_lucas_kanade.py` | Lucas-Kanade sparse flow |
| 02 | `02_dense_optical_flow_farneback.py` | Farneback dense flow |
| 03 | `03_visualisasi_optical_flow.py` | Visualisasi flow vectors |
| 04 | `04_background_subtraction_mog2.py` | MOG2 background subtraction |
| 05 | `05_background_subtraction_knn.py` | KNN background subtraction |
| 06 | `06_frame_differencing.py` | Frame differencing |
| 07 | `07_running_average_background.py` | Running average BG model |
| 08 | `08_object_tracking_csrt.py` | CSRT tracker |
| 09 | `09_object_tracking_kcf.py` | KCF tracker |
| 10 | `10_multi_object_tracking.py` | Multi-object tracking |
| 11 | `11_motion_history_image.py` | Motion History Image (MHI) |
| 12 | `12_video_stabilization.py` | Video stabilization |
| 13 | `13_frame_interpolation_linear.py` | Linear frame interpolation |
| 14 | `14_frame_interpolation_flow.py` | Flow-based interpolation |
| 15 | `15_optical_flow_magnitude_direction.py` | Magnitude & direction |
| 16 | `16_feature_trajectory_tracking.py` | Feature trajectory |
| 17 | `17_background_subtraction_comparison.py` | Perbandingan BG subtraction |
| 18 | `18_deteksi_gerakan_contour.py` | Deteksi gerakan via kontur |
| 19 | `19_optical_flow_realtime_simulasi.py` | Simulasi real-time flow |
| 20 | `20_estimasi_kecepatan_objek.py` | Estimasi kecepatan objek |

</details>

<details>
<summary><strong>Modul 10 — Computational Photography</strong></summary>

> **Referensi:** Szeliski Ch.10 | **Dokumen:** [Materi](Modul10%20-%20Computational%20Photography/Materi.md) · [Jobsheet](Modul10%20-%20Computational%20Photography/Jobsheet.md) · [Project](Modul10%20-%20Computational%20Photography/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_hdr_imaging_pipeline.py` | Pipeline HDR imaging |
| 02 | `02_tone_mapping_reinhard.py` | Tone mapping Reinhard |
| 03 | `03_tone_mapping_drago.py` | Tone mapping Drago |
| 04 | `04_exposure_fusion_mertens.py` | Exposure fusion Mertens |
| 05 | `05_denoising_gaussian_blur.py` | Denoising Gaussian |
| 06 | `06_denoising_bilateral_filter.py` | Denoising bilateral |
| 07 | `07_denoising_non_local_means.py` | Non-local means denoising |
| 08 | `08_image_inpainting_ns.py` | Inpainting Navier-Stokes |
| 09 | `09_image_inpainting_telea.py` | Inpainting Telea |
| 10 | `10_super_resolution_interpolasi.py` | Super-resolution dasar |
| 11 | `11_clahe_enhancement.py` | CLAHE enhancement |
| 12 | `12_unsharp_mask_sharpening.py` | Unsharp mask |
| 13 | `13_white_balance_correction.py` | White balance |
| 14 | `14_synthetic_bokeh_effect.py` | Bokeh efek sintetis |
| 15 | `15_color_enhancement_saturation.py` | Saturasi warna |
| 16 | `16_image_enhancement_pipeline.py` | Enhancement pipeline |
| 17 | `17_pencil_sketch_effect.py` | Efek sketsa pensil |
| 18 | `18_cartoon_effect_stylization.py` | Efek kartun |
| 19 | `19_hdr_dari_single_image.py` | Pseudo-HDR single image |
| 20 | `20_style_transfer_manual.py` | Style transfer manual |

</details>

<details>
<summary><strong>Modul 11 — Structure from Motion dan Depth Estimation</strong></summary>

> **Referensi:** Szeliski Ch.11–12 | **Dokumen:** [Materi](Modul11%20-%20Structure%20from%20Motion%20dan%20Depth%20Estimation/Materi.md) · [Jobsheet](Modul11%20-%20Structure%20from%20Motion%20dan%20Depth%20Estimation/Jobsheet.md) · [Project](Modul11%20-%20Structure%20from%20Motion%20dan%20Depth%20Estimation/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_deteksi_fitur_dan_matching.py` | Fitur & matching untuk SfM |
| 02 | `02_fundamental_matrix.py` | Estimasi Fundamental Matrix |
| 03 | `03_essential_matrix_pose.py` | Essential Matrix & pose |
| 04 | `04_epipolar_lines.py` | Epipolar geometry |
| 05 | `05_triangulasi_titik_3d.py` | Triangulasi titik 3D |
| 06 | `06_stereo_calibration.py` | Kalibrasi stereo kamera |
| 07 | `07_stereo_rectification.py` | Rektifikasi stereo |
| 08 | `08_block_matching_disparity.py` | Block matching disparity |
| 09 | `09_sgbm_disparity.py` | Semi-global block matching |
| 10 | `10_monocular_depth_estimation.py` | Monocular depth |
| 11 | `11_disparity_to_depth.py` | Disparity → depth |
| 12 | `12_stereo_bm_vs_sgbm.py` | BM vs SGBM perbandingan |
| 13 | `13_wls_filter_disparity.py` | WLS filter disparity |
| 14 | `14_point_cloud_from_depth.py` | Point cloud dari depth |
| 15 | `15_pnp_pose_estimation.py` | PnP pose estimation |
| 16 | `16_stereo_matching_realtime.py` | Stereo matching real-time |
| 17 | `17_depth_map_colorization.py` | Kolorisasi depth map |
| 18 | `18_baseline_effect_depth.py` | Efek baseline pada depth |
| 19 | `19_depth_object_segmentation.py` | Segmentasi berbasis depth |
| 20 | `20_multiview_reconstruction_pipeline.py` | Pipeline multiview |

</details>

<details>
<summary><strong>Modul 12 — Rekonstruksi 3D dan Image-Based Rendering</strong></summary>

> **Referensi:** Szeliski Ch.13–14 | **Dokumen:** [Materi](Modul12%20-%20Rekonstruksi%203D%20dan%20Image-Based%20Rendering/Materi.md) · [Jobsheet](Modul12%20-%20Rekonstruksi%203D%20dan%20Image-Based%20Rendering/Jobsheet.md) · [Project](Modul12%20-%20Rekonstruksi%203D%20dan%20Image-Based%20Rendering/Project.md)

**Program Praktikum:**
| No | File | Topik |
|----|------|-------|
| 01 | `01_point_cloud_basics.py` | Dasar point cloud Open3D |
| 02 | `02_point_cloud_filtering.py` | Filter & downsample |
| 03 | `03_normal_estimation.py` | Estimasi normal |
| 04 | `04_icp_registration.py` | ICP registration |
| 05 | `05_surface_reconstruction.py` | Rekonstruksi permukaan |
| 06 | `06_mesh_processing.py` | Pemrosesan mesh |
| 07 | `07_tsdf_integration.py` | TSDF volumetric fusion |
| 08 | `08_image_warping_depth.py` | Image warping berbasis depth |
| 09 | `09_view_interpolation.py` | View interpolation |
| 10 | `10_neural_rendering_intro.py` | Intro neural rendering |
| 11 | `11_ball_pivoting_algorithm.py` | Ball pivoting |
| 12 | `12_alpha_shapes.py` | Alpha shapes |
| 13 | `13_mesh_texturing.py` | Texturing pada mesh |
| 14 | `14_point_cloud_colorization.py` | Kolorisasi point cloud |
| 15 | `15_point_cloud_segmentation.py` | Segmentasi point cloud |
| 16 | `16_marching_cubes.py` | Marching cubes |
| 17 | `17_volumetric_rendering.py` | Volumetric rendering |
| 18 | `18_forward_inverse_warping.py` | Forward & inverse warping |
| 19 | `19_light_field_basics.py` | Light field dasar |
| 20 | `20_3d_visualization_export.py` | Export 3D ke berbagai format |

</details>

---

## ⚙️ Instalasi & Setup

### Prasyarat

- Python **3.8+**
- RAM minimal **4 GB** (8 GB+ direkomendasikan untuk modul 11–12)
- GPU opsional (mempercepat modul 05)

### Langkah Instalasi

```bash
# 1. Clone repo
git clone https://github.com/rofiqcp/Praktikum-Komputer-Vision.git
cd Praktikum-Komputer-Vision

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktifkan virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Install semua dependency
pip install -r requirements.txt
```

### Library Utama yang Digunakan

| Library | Versi | Kegunaan |
|---------|-------|----------|
| `opencv-python` | ≥ 4.8 | Library utama CV |
| `opencv-contrib-python` | ≥ 4.8 | SIFT, AKAZE, extras |
| `numpy` | ≥ 1.24 | Operasi array & matriks |
| `matplotlib` | ≥ 3.10 | Visualisasi & plotting |
| `torch` / `torchvision` | ≥ 2.0 | Deep learning (modul 05) |
| `scikit-image` | ≥ 0.21 | Algoritma image processing |
| `open3d` | ≥ 0.17 | Point cloud & mesh (modul 11–12) |
| `scipy` | ≥ 1.11 | Optimasi & matematika |
| `Pillow` | ≥ 10.0 | I/O gambar |

---

## ▶️ Cara Belajar yang Disarankan

```
Baca Materi.md  →  Ikuti Jobsheet.md  →  Jalankan program di praktikum/  →  Kerjakan Project.md
```

1. **Baca [Materi.md]** — Pahami teori di balik setiap topik
2. **Ikuti [Jobsheet.md]** — Langkah demi langkah praktikum terstruktur
3. **Jalankan Python** — Eksekusi setiap file di folder `praktikum/` dan amati output
4. **Kerjakan [Project.md]** — Integrasikan konsep dalam proyek mingguan
5. **Rekam [TugasVideo.md]** — Presentasikan hasil dalam format video
6. **Baca [review/](review/)** — Cermati analisis kelengkapan per modul

### Jalur Belajar

```
Modul 01 (Dasar)
    ↓
Modul 02 (Kamera & Geometri)
    ↓
Modul 03 (Pemrosesan)  →  Modul 04 (Fitting)
    ↓                              ↓
Modul 05 (Deep Learning)  ←  Modul 06 (Recognition)
    ↓
Modul 07 (Fitur) → Modul 08 (Stitching)
                            ↓
              Modul 09 (Motion) → Modul 10 (Photo)
                                          ↓
                            Modul 11 (SfM & Depth)
                                          ↓
                           Modul 12 (3D & Rendering)
```

---

## 🧭 Struktur Repo

```
Praktikum-Komputer-Vision/
├── README.md                          # Dokumen ini
├── requirements.txt                   # Semua dependency Python
│
├── Modul01 - Pendahuluan Komputer Vision/
│   ├── Materi.md                      # Teori & konsep
│   ├── Jobsheet.md                    # Panduan langkah demi langkah
│   ├── Project.md                     # Proyek integrasi
│   ├── TugasVideo.md                  # Panduan tugas video
│   ├── notebookllm.md                 # Catatan AI/LLM
│   └── praktikum/
│       ├── 01_*.py … 20_*.py          # 20 program terstruktur
│       └── download_image.py          # Helper download gambar
│
├── Modul02 … Modul12/                 # Struktur identik
│
├── Referensi/
│   ├── -Computer Vision- Algorithms and Applications 2nd Edition.pdf
│   ├── Bab-01-Introduction.pdf … Bab-14-Image-based-rendering.pdf
│   └── other/                         # 15 buku referensi tambahan
│
└── review/
    └── review1.md … review9.md        # Analisis & review per modul
```

---

## 📖 Referensi

### Buku Utama

| Bab | Judul | File |
|-----|-------|------|
| — | **Computer Vision: Algorithms and Applications 2nd Ed.** (Szeliski) | [PDF](Referensi/-Computer%20Vision-%20Algorithms%20and%20Applications%202nd%20Edition%2C%20Richard%20Szeliski.pdf) |
| 01 | Introduction | [Bab-01](Referensi/Bab-01-Introduction.pdf) |
| 02 | Image Formation | [Bab-02](Referensi/Bab-02-Image%20formation.pdf) |
| 03 | Image Processing | [Bab-03](Referensi/Bab-03-Image%20processing.pdf) |
| 04 | Model Fitting | [Bab-04](Referensi/Bab-04-Model%20fitting%20and%20optimization.pdf) |
| 05 | Deep Learning | [Bab-05](Referensi/Bab-05-Deep%20Learning.pdf) |
| 06 | Recognition | [Bab-06](Referensi/Bab-06-Recognition.pdf) |
| 07 | Feature Detection | [Bab-07](Referensi/Bab-07-Feature%20detection%20and%20matching.pdf) |
| 08 | Alignment & Stitching | [Bab-08](Referensi/Bab-08-Image%20alignment%20and%20stitching.pdf) |
| 09 | Motion Estimation | [Bab-09](Referensi/Bab-09-Motion%20estimation.pdf) |
| 10 | Computational Photography | [Bab-10](Referensi/Bab-10-Computational%20photography.pdf) |
| 11 | Structure from Motion | [Bab-11](Referensi/Bab-11-Structure%20from%20motion%20and%20SLAM.pdf) |
| 12 | Depth Estimation | [Bab-12](Referensi/Bab-12-Depth%20estimation.pdf) |
| 13 | 3D Reconstruction | [Bab-13](Referensi/Bab-13-3D%20reconstruction.pdf) |
| 14 | Image-Based Rendering | [Bab-14](Referensi/Bab-14-Image-based-rendering.pdf) |

### Referensi Tambahan (folder `Referensi/other/`)

- Learning OpenCV (O'Reilly) · Mastering OpenCV 3 & 4 · Practical Intro to CV with OpenCV
- Deep Learning for Computer Vision (Python) · Machine Learning for OpenCV
- Hands-on ML with Scikit-Learn, Keras & TensorFlow (Géron, 2 edisi)
- Hands-on ML Projects with OpenCV · TensorFlow Workshop

---

## 🎓 Target Capaian Pembelajaran

Setelah menyelesaikan seluruh modul, mahasiswa mampu:

- [ ] Memahami konsep inti computer vision dari citra 2D hingga rekonstruksi 3D
- [ ] Mengimplementasikan algoritma klasik (filtering, matching, stitching, tracking)
- [ ] Merancang dan melatih model deep learning untuk deteksi & segmentasi
- [ ] Membangun pipeline SfM, stereo depth, dan 3D reconstruction
- [ ] Menerapkan teknik computational photography (HDR, denoising, inpainting)
- [ ] Mengevaluasi performa model dengan metrik standar industri
- [ ] Mengembangkan aplikasi computer vision end-to-end

---

## 📝 Review & Analisis Modul

Folder [`review/`](review/) berisi analisis mendalam setiap modul mencakup:
- Kelengkapan materi vs. referensi buku
- Saran perbaikan dan penambahan topik
- Identifikasi gap dan potensi pengembangan

| Review | Isi |
|--------|-----|
| [review1.md](review/review1.md) | Analisis Modul 01 |
| [review2.md](review/review2.md) | Analisis Modul 02 |
| [review3.md](review/review3.md) | Analisis Modul 03 |
| [review4.md](review/review4.md) | Analisis Modul 04 |
| [review5.md](review/review5.md) | Analisis Modul 05 |
| [review6.md](review/review6.md) | Analisis Modul 06 |
| [review7.md](review/review7.md) | Analisis Modul 07 |
| [review8.md](review/review8.md) | Analisis Modul 08 |
| [review9.md](review/review9.md) | Analisis Modul 09 |

---

## 🙌 Kontribusi

Kontribusi sangat terbuka! Anda dapat:
- Memperbaiki program yang belum optimal
- Menambahkan program baru ke modul yang ada
- Melengkapi dokumentasi Materi/Jobsheet
- Menambahkan contoh output atau visualisasi
- Membuat review untuk modul 10–12

---

<div align="center">

**Status Repo:** Aktif dan terus disempurnakan

Dibuat dengan semangat belajar Computer Vision · Berbasis **Szeliski CV 2nd Ed.**

</div>
