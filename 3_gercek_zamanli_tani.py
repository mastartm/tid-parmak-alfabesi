# -*- coding: utf-8 -*-
"""
ADIM 3 (GUNCEL): GERCEK ZAMANLI TANIMA - iki el destegi

Yeni veri formatiyla (84 ozellik) calisir.
Kullanim: python 3_gercek_zamanli_tani.py
C: metni temizle | Q: cik
"""

import pickle
import time

import cv2
import mediapipe as mp
import numpy as np

MODEL_DOSYASI = "model.pkl"
SABIT_KARE_ESIGI = 25
GUVEN_ESIGI = 0.6
EL_YOK_BOSLUK_SURESI = 1.5

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def tek_el_normalize(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    coords -= coords[0]
    max_deger = np.abs(coords).max()
    if max_deger > 0:
        coords /= max_deger
    return coords.flatten()


def iki_el_ozellik_vektoru(multi_hand_landmarks):
    vektor = np.zeros(84)
    eller = list(multi_hand_landmarks)[:2]
    eller.sort(key=lambda el: el.landmark[0].x)
    for i, el in enumerate(eller):
        vektor[i * 42:(i + 1) * 42] = tek_el_normalize(el)
    return vektor


def main():
    with open(MODEL_DOSYASI, "rb") as f:
        model = pickle.load(f)
    print("Model yuklendi. Kamera aciliyor...")

    kamera = cv2.VideoCapture(0)

    metin = ""
    aday_harf = None
    aday_sayac = 0
    son_el_zamani = time.time()
    bosluk_eklendi = True

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, kare = kamera.read()
            if not ok:
                break

            kare = cv2.flip(kare, 1)
            rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            sonuc = hands.process(rgb)

            anlik_harf, guven = None, 0.0

            if sonuc.multi_hand_landmarks:
                son_el_zamani = time.time()
                bosluk_eklendi = False
                for el in sonuc.multi_hand_landmarks:
                    mp_draw.draw_landmarks(kare, el, mp_hands.HAND_CONNECTIONS)

                veri = iki_el_ozellik_vektoru(
                    sonuc.multi_hand_landmarks).reshape(1, -1)
                olasiliklar = model.predict_proba(veri)[0]
                en_iyi = np.argmax(olasiliklar)
                guven = olasiliklar[en_iyi]
                if guven >= GUVEN_ESIGI:
                    anlik_harf = model.classes_[en_iyi]
            else:
                if (not bosluk_eklendi and metin and not metin.endswith(" ")
                        and time.time() - son_el_zamani > EL_YOK_BOSLUK_SURESI):
                    metin += " "
                    bosluk_eklendi = True

            if anlik_harf is not None and anlik_harf == aday_harf:
                aday_sayac += 1
                if aday_sayac == SABIT_KARE_ESIGI:
                    metin += str(anlik_harf)
            else:
                aday_harf = anlik_harf
                aday_sayac = 0

            if anlik_harf is not None:
                ilerleme = min(aday_sayac / SABIT_KARE_ESIGI, 1.0)
                cv2.putText(kare, f"{anlik_harf} (%{guven*100:.0f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.rectangle(kare, (10, 55), (10 + int(200 * ilerleme), 70),
                              (0, 255, 0), -1)
                cv2.rectangle(kare, (10, 55), (210, 70), (255, 255, 255), 2)

            h = kare.shape[0]
            cv2.rectangle(kare, (0, h - 60), (kare.shape[1], h), (0, 0, 0), -1)
            cv2.putText(kare, "Metin: " + metin[-40:], (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Isaret Dili Tanima - Q: cik, C: temizle", kare)
            tus = cv2.waitKey(1) & 0xFF
            if tus == ord("q"):
                break
            elif tus == ord("c"):
                metin = ""

    kamera.release()
    cv2.destroyAllWindows()
    print("\nOlusturulan metin:", metin)


if __name__ == "__main__":
    main()
