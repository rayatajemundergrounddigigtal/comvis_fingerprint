# ============================================================
# preprocessing.py
# Tahap 1 & 2 flow: INPUT CITRA + PREPROCESSING & ENHANCEMENT
# Materi: LS 1 (I/O citra), LS 2 (grayscale & resize),
#         LS 3 (CLAHE), LS 4 (filter spasial & unsharp masking)
# ============================================================

import cv2
import numpy as np

UKURAN_STANDAR = (400, 400)  # (lebar, tinggi) — semua citra disamakan ukurannya


def load_grayscale(path, size=UKURAN_STANDAR):
    """[LS 1 & LS 2] Baca citra lalu konversi ke grayscale + resize.

    Kenapa grayscale? Warna tidak mengandung informasi pola sidik jari,
    yang penting hanya gelap-terang garis (ridge/valley).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Citra tidak ditemukan: {path}")
    return cv2.resize(img, size)


def enhance(gray, clip_limit=3.0, tile_grid=(8, 8), blur_ksize=5, unsharp_k=0.0):
    """[LS 3 & LS 4] Pipeline enhancement.

    Urutan:
      1. CLAHE        -> perbaiki kontras LOKAL per tile 8x8 (LS 3)
                         clipLimit membatasi penguatan supaya noise tidak ikut kuat
      2. GaussianBlur -> redam noise halus sebelum deteksi fitur (LS 4)
                         kernel ganjil supaya punya titik pusat yang jelas
      3. Unsharp Masking (opsional) -> tajamkan ridge: asli + k*(asli - blur) (LS 4)

    Mengembalikan dict semua tahapan supaya bisa ditampilkan saat demo.
    """
    # --- LS 3: CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    hasil_clahe = clahe.apply(gray)

    # --- LS 4: Gaussian Blur (kernel harus ganjil) ---
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    hasil_blur = cv2.GaussianBlur(hasil_clahe, (k, k), 0)

    # --- LS 4: Unsharp Masking (persis praktik 4.3) ---
    if unsharp_k > 0:
        blurred = cv2.GaussianBlur(hasil_blur, (9, 9), 10.0)
        mask_detail = cv2.subtract(hasil_blur, blurred)
        hasil_akhir = cv2.addWeighted(hasil_blur, 1.0, mask_detail, unsharp_k, 0)
    else:
        hasil_akhir = hasil_blur

    return {
        "asli": gray,
        "clahe": hasil_clahe,
        "blur": hasil_blur,
        "enhanced": hasil_akhir,  # <- citra INI yang dipakai untuk matching ORB
    }


def tambah_noise_salt_pepper(image, salt_prob=0.02, pepper_prob=0.02):
    """[LS 4] Pembangkit noise salt-and-pepper (persis praktik 4.1).
    Dipakai untuk menguji ketahanan sistem terhadap citra kotor.
    """
    noisy = image.copy()
    num_salt = int(np.ceil(salt_prob * image.size))
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[tuple(coords)] = 255
    num_pepper = int(np.ceil(pepper_prob * image.size))
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[tuple(coords)] = 0
    return noisy


def tambah_noise_gaussian(image, mean=0, var=0.02):
    """[LS 4] Pembangkit noise Gaussian (persis praktik 4.1)."""
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape) * 255
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)