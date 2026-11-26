# ============================================================
# matcher.py
# Tahap 5 & 6 flow: EKSTRAKSI FITUR, PENCOCOKAN & KEPUTUSAN
# Materi BARU (di luar labsheet): ORB + BFMatcher + Ratio Test
#
# Analogi menjelaskan ke dosen:
# "ORB menandai landmark unik di dua peta (pertigaan, bundaran),
#  lalu mendeskripsikan bentuk sekitarnya. Matching = menghitung
#  berapa landmark yang sama. Makin banyak yang cocok, makin
#  mirip kedua sidik jari."
# ============================================================

import cv2

RATIO_TEST = 0.75  # Lowe's ratio test: kandidat terbaik harus jauh lebih
                   # baik dari kandidat kedua, kalau tidak -> match ambigu, buang


def ekstrak_fitur(img_enhanced, nfeatures=1000):
    """Deteksi keypoint + descriptor dengan ORB.

    Kenapa ORB bukan SIFT? ORB lebih cepat, gratis, dan akurasinya
    cukup untuk dataset sidik jari ukuran ini.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img_enhanced, None)
    return keypoints, descriptors


def cocokkan(img1_enhanced, img2_enhanced, ratio=RATIO_TEST):
    """Bandingkan dua citra enhanced (bukan biner!).

    Mengembalikan dict:
      skor        : persentase kemiripan 0-100
      good_matches: daftar match lolos ratio test
      keypoints   : keypoint kedua citra (untuk visualisasi drawMatches)
      gambar_match: visualisasi garis penghubung titik-titik yang cocok

    Rumus skor (mudah dijelaskan):
      skor = good_matches / min(jumlah keypoint citra 1, citra 2) x 100%
    """
    kp1, des1 = ekstrak_fitur(img1_enhanced)
    kp2, des2 = ekstrak_fitur(img2_enhanced)

    hasil_kosong = {
        "skor": 0.0, "good_matches": [], "kp1": kp1, "kp2": kp2,
        "gambar_match": None, "jumlah_kp": (len(kp1), len(kp2)),
    }

    # Guard: kalau descriptor kosong / keypoint terlalu sedikit
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return hasil_kosong

    # BFMatcher dengan NORM_HAMMING (descriptor ORB berbentuk biner)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pasangan = bf.knnMatch(des1, des2, k=2)

    # Ratio test: buang match yang ambigu
    good = []
    for p in pasangan:
        if len(p) == 2:
            m, n = p
            if m.distance < ratio * n.distance:
                good.append(m)

    skor = len(good) / min(len(kp1), len(kp2)) * 100.0

    # Visualisasi: garis hijau menghubungkan titik-titik yang cocok
    gambar_match = cv2.drawMatches(
        img1_enhanced, kp1, img2_enhanced, kp2, good, None,
        matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )

    return {
        "skor": skor,
        "good_matches": good,
        "kp1": kp1, "kp2": kp2,
        "gambar_match": gambar_match,
        "jumlah_kp": (len(kp1), len(kp2)),
    }


def keputusan(skor, threshold=8.0):
    """Tahap 6: COCOK / TIDAK COCOK.

    PENTING: threshold TIDAK boleh asal tebak (mis. langsung 75%).
    Nilai wajar bergantung pada dataset & jumlah keypoint — harus
    dikalibrasi lewat evaluate.py --sweep.

    Threshold = trade-off keamanan vs kenyamanan:
    - terlalu tinggi -> jari asli sering ditolak (FRR naik, tidak nyaman)
    - terlalu rendah -> jari orang lain bisa lolos (FAR naik, tidak aman)
    """
    return skor >= threshold