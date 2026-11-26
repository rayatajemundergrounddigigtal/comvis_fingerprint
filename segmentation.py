# ============================================================
# segmentation.py
# Tahap 3 & 4 flow: SEGMENTASI + OPERASI MORFOLOGI
# Materi: LS 7 (Otsu & Adaptive Threshold), LS 6 (Opening/Closing)
#
# CATATAN PENTING (poin analisis untuk laporan):
# Hasil biner di sini dipakai untuk VISUALISASI tahapan dan
# segmentasi area jari — BUKAN untuk matching ORB.
# Alasan: citra biner terlalu kasar, banyak sudut palsu yang
# mirip semua, sehingga descriptor ORB jadi tidak diskriminatif.
# ============================================================

import cv2


def binarisasi(gray, metode="adaptive", block_size=11, c=2):
    """[LS 7] Ubah grayscale menjadi citra biner (hitam-putih).

    - otsu     : threshold otomatis dari histogram (satu nilai global)
    - adaptive : threshold berbeda per area — cocok untuk sidik jari
                 karena tekanan jari tidak merata (tengah lebih gelap
                 daripada pinggir). Persis praktik 7.2.
    """
    if metode == "otsu":
        val, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary, val
    else:
        bs = block_size if block_size % 2 == 1 else block_size + 1
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, bs, c
        )
        return binary, None


def bersihkan_morfologi(binary, ksize=3):
    """[LS 6] Rantai pembersihan masker biner.

    Urutan WAJIB: Opening dulu, baru Closing.
    - Opening (erosi->dilasi) : hapus bintik noise kecil ("mengayak pasir")
    - Closing (dilasi->erosi) : sambung ridge yang putus ("menambal aspal")

    Kalau closing duluan, bintik noise malah membesar (lihat
    Keselamatan Kerja LS 6).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return {"opening": opening, "clean": closing}


def segmentasi_lengkap(gray, metode="adaptive"):
    """Gabungan tahap 3 + 4, mengembalikan semua tahapan untuk panel demo."""
    binary, nilai_otsu = binarisasi(gray, metode=metode)
    morph = bersihkan_morfologi(binary)
    return {
        "biner": binary,
        "opening": morph["opening"],
        "morfologi": morph["clean"],
        "nilai_otsu": nilai_otsu,
    }