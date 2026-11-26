# ============================================================
# generate_samples.py
# Membuat citra sidik jari SINTETIS untuk latihan/testing awal
# sebelum memakai dataset asli SOCOFing.
#
# Pola ridge dibuat dari fungsi sinus melingkar yang di-warp
# secara acak — tiap "jari" (seed) punya pola unik.
#
# Jalankan:  python3 generate_samples.py
# Hasil di folder samples/:
#   jari_A.png          -> jari A asli
#   jari_A_rotasi.png   -> jari A diputar 15 derajat (uji rotasi)
#   jari_A_noise.png    -> jari A + noise salt-pepper & gaussian (LS 4)
#   jari_A_blur.png     -> jari A kabur (uji enhancement)
#   jari_B.png, jari_C.png -> jari orang lain (harus DITOLAK)
# ============================================================

import os

import cv2
import numpy as np

from preprocessing import tambah_noise_salt_pepper, tambah_noise_gaussian

UKURAN = 400


def buat_sidik_jari(seed, size=UKURAN):
    """Buat pola ridge unik berdasarkan seed (deterministik)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)

    # Posisi core (pusat pusaran) sedikit acak per jari
    cx = size * 0.5 + rng.uniform(-25, 25)
    cy = size * 0.45 + rng.uniform(-25, 25)
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    # Parameter unik per jari
    freq = rng.uniform(0.55, 0.75)          # kerapatan ridge
    fasa = rng.uniform(0, 2 * np.pi)
    warp_amp = rng.uniform(4, 10)           # lengkungan pola
    warp_freq = rng.integers(2, 5)

    # Warp halus supaya pola tidak sempurna melingkar (lebih realistis)
    warp = warp_amp * np.sin(warp_freq * theta + fasa)
    warp += 6 * np.sin(x * 0.02 + fasa) + 6 * np.cos(y * 0.025 + seed)

    # Medan warp acak halus (unik per seed) -> struktur lokal khas,
    # mirip minutiae, supaya keypoint ORB punya ciri yang diskriminatif
    grid = rng.uniform(-1, 1, (10, 10))
    field = cv2.resize(grid.astype(np.float32), (size, size),
                       interpolation=cv2.INTER_CUBIC)
    warp += field * rng.uniform(10, 16)

    ridge = np.sin((r + warp) * freq)
    img = (128 + 110 * ridge).astype(np.uint8)

    # "Minutiae" sintetis: potong ridge secara acak (ridge ending)
    # dan tambah titik gelap (bifurcation) -> struktur lokal unik per jari,
    # membuat keypoint ORB jauh lebih diskriminatif
    for _ in range(180):
        px, py = rng.integers(40, size - 40, 2)
        panjang = int(rng.integers(4, 9))
        sudut = rng.uniform(0, 2 * np.pi)
        dx2, dy2 = int(panjang * np.cos(sudut)), int(panjang * np.sin(sudut))
        cv2.line(img, (int(px), int(py)), (int(px) + dx2, int(py) + dy2),
                 235, 2)  # potong ridge dengan warna valley
    for _ in range(80):
        px, py = rng.integers(40, size - 40, 2)
        cv2.circle(img, (int(px), int(py)), int(rng.integers(1, 3)), 30, -1)

    # Masker elips: area jari saja, background putih (seperti scan jari)
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (int(size * 0.38), int(size * 0.45)),
                0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    alpha = mask.astype(np.float64) / 255.0
    img = (img * alpha + 235 * (1 - alpha)).astype(np.uint8)
    return img


def rotasi(img, derajat):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), derajat, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=235)


def utama():
    os.makedirs("samples", exist_ok=True)
    a = buat_sidik_jari(seed=11)
    b = buat_sidik_jari(seed=77)
    c = buat_sidik_jari(seed=123)

    cv2.imwrite("samples/jari_A.png", a)
    cv2.imwrite("samples/jari_A_rotasi.png", rotasi(a, 15))

    noisy = tambah_noise_salt_pepper(a, 0.02, 0.02)   # LS 4
    noisy = tambah_noise_gaussian(noisy, var=0.02)    # LS 4
    cv2.imwrite("samples/jari_A_noise.png", noisy)

    cv2.imwrite("samples/jari_A_blur.png", cv2.GaussianBlur(a, (9, 9), 3))
    cv2.imwrite("samples/jari_B.png", b)
    cv2.imwrite("samples/jari_C.png", c)
    cv2.imwrite("samples/jari_asing.png", buat_sidik_jari(seed=999))

    # Database "pengguna terdaftar" untuk mode identification (1-lawan-N).
    # Nama file = nama pemilik, supaya banner demo menampilkan nama orang.
    os.makedirs("database", exist_ok=True)
    cv2.imwrite("database/Andi.png", a)
    cv2.imwrite("database/Budi.png", b)
    cv2.imwrite("database/Citra.png", c)
    print("Sampel berhasil dibuat di folder samples/ (7 file).")
    print("Database pengguna terdaftar: database/ (Andi, Budi, Citra).")


if __name__ == "__main__":
    utama()