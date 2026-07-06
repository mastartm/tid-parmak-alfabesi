# -*- coding: utf-8 -*-
"""
ADIM 1 (GUNCEL): VERI TOPLAMA - Otomatik kayit + iki el destegi

Kullanim:
    python 1_veri_topla.py

- Harfi yazin (orn: A), kamera acilir
- BOSLUK'a BIR KEZ basin -> 3 saniye geri sayim -> otomatik kayit baslar
  (Elleriniz tamamen serbest, klavyeye dokunmaniza gerek yok)
- Kayit sirasinda elinizi hafifce oynatin: yakin/uzak, hafif acilar
- Hedef ornek sayisina ulasinca kayit KENDILIGINDEN durur
  (istediginiz an BOSLUK ile de durdurabilirsiniz)
- Q ile cikin

NOT: Bu surum IKI ELI destekler. Tek elli harflerde ikinci el
     alani sifirlarla doldurulur - sorun degil, model bunu ogrenir.

ONEMLI: Daha once eski surumle veri topladiysaniz isaret_verisi.csv
        dosyasini SILIN - veri formati degisti (42 -> 84 sutun).
"""

import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np

VERI_DOSYASI = "isaret_verisi.csv"
HEDEF_ORNEK = 200          # bu sayiya ulasinca kayit otomatik durur
GERI_SAYIM_SANIYE = 3      # kayit baslamadan once hazirlanma suresi

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def tek_el_normalize(landmarks):
    """Bir elin 21 noktasini bilege gore normalize eder -> 42 deger."""
    coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    coords -= coords[0]
    max_deger = np.abs(coords).max()
    if max_deger > 0:
        coords /= max_deger
    return coords.flatten()


def iki_el_ozellik_vektoru(multi_hand_landmarks):
    """
    Iki el icin 84 degerlik sabit boyutlu vektor uretir.
    Eller ekrandaki yatay konumlarina gore siralanir (sol slot, sag slot)
    ki ayni isaret hep ayni sirayla kaydedilsin.
    Eksik el sifirlarla doldurulur.
    """
    vektor = np.zeros(84)
    eller = list(multi_hand_landmarks)[:2]
    # Bilek x koordinatina gore sirala (soldaki el once)
    eller.sort(key=lambda el: el.landmark[0].x)
    for i, el in enumerate(eller):
        vektor[i * 42:(i + 1) * 42] = tek_el_normalize(el)
    return vektor.tolist()


def main():
    harf = input("Hangi harf icin veri toplanacak? (orn: A): ").strip().upper()
    if not harf:
        print("Gecersiz harf.")
        return

    dosya_var = os.path.exists(VERI_DOSYASI)
    f = open(VERI_DOSYASI, "a", newline="", encoding="utf-8")
    yazici = csv.writer(f)
    if not dosya_var:
        baslik = ["harf"] + [f"el{e}_{eksen}{i}"
                             for e in (1, 2)
                             for i in range(21)
                             for eksen in ("x", "y")]
        yazici.writerow(baslik)

    kamera = cv2.VideoCapture(0)
    sayac = 0
    durum = "bekleme"          # bekleme -> gerisayim -> kayit
    gerisayim_baslangic = 0

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        print(f"\n'{harf}' harfi icin hazir.")
        print("BOSLUK = kayit baslat/durdur | Q = cik\n")

        while True:
            ok, kare = kamera.read()
            if not ok:
                print("Kamera okunamadi!")
                break

            kare = cv2.flip(kare, 1)
            rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            sonuc = hands.process(rgb)

            el_var = bool(sonuc.multi_hand_landmarks)
            if el_var:
                for el in sonuc.multi_hand_landmarks:
                    mp_draw.draw_landmarks(kare, el, mp_hands.HAND_CONNECTIONS)

            # ---- Durum makinesi ----
            if durum == "gerisayim":
                kalan = GERI_SAYIM_SANIYE - (time.time() - gerisayim_baslangic)
                if kalan <= 0:
                    durum = "kayit"
                else:
                    cv2.putText(kare, str(int(kalan) + 1),
                                (kare.shape[1] // 2 - 40, kare.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 165, 255), 8)

            if durum == "kayit" and el_var:
                veri = iki_el_ozellik_vektoru(sonuc.multi_hand_landmarks)
                yazici.writerow([harf] + veri)
                sayac += 1
                if sayac >= HEDEF_ORNEK:
                    durum = "bekleme"
                    print(f"Hedefe ulasildi: {sayac} ornek. "
                          f"Devam icin tekrar BOSLUK'a basabilirsiniz.")

            # ---- Ekran bilgileri ----
            renk = (0, 0, 255) if durum == "kayit" else (0, 255, 0)
            durum_yazi = {"bekleme": "HAZIR (BOSLUK: baslat)",
                          "gerisayim": "HAZIRLANIN...",
                          "kayit": "KAYIT YAPILIYOR"}[durum]
            cv2.putText(kare, f"Harf: {harf}  Ornek: {sayac}/{HEDEF_ORNEK}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, renk, 2)
            cv2.putText(kare, durum_yazi, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)
            if durum == "kayit" and not el_var:
                cv2.putText(kare, "El algilanmadi - bekleniyor", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Veri Toplama", kare)
            tus = cv2.waitKey(1) & 0xFF

            if tus == ord(" "):
                if durum == "bekleme":
                    durum = "gerisayim"
                    gerisayim_baslangic = time.time()
                else:
                    durum = "bekleme"
            elif tus == ord("q"):
                break

    f.close()
    kamera.release()
    cv2.destroyAllWindows()
    print(f"\nToplam {sayac} ornek '{harf}' harfi icin kaydedildi -> {VERI_DOSYASI}")


if __name__ == "__main__":
    main()
