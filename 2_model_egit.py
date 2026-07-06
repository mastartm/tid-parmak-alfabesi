# -*- coding: utf-8 -*-
"""
ADIM 2: MODEL EGITIMI
Toplanan CSV verisiyle bir Random Forest siniflandirici egitir.

Kullanim:
    python 2_model_egit.py

Cikti: model.pkl (egitilmis model dosyasi)
"""

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

VERI_DOSYASI = "isaret_verisi.csv"
MODEL_DOSYASI = "model.pkl"


def main():
    print("Veri okunuyor...")
    df = pd.read_csv(VERI_DOSYASI, encoding="utf-8")

    print(f"Toplam ornek: {len(df)}")
    print("Harf basina ornek sayilari:")
    print(df["harf"].value_counts().to_string())

    X = df.drop(columns=["harf"]).values
    y = df["harf"].values

    if len(set(y)) < 2:
        print("\nUYARI: En az 2 farkli harf icin veri toplamalisiniz!")
        return

    X_egitim, X_test, y_egitim, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nModel egitiliyor...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_egitim, y_egitim)

    tahmin = model.predict(X_test)
    dogruluk = accuracy_score(y_test, tahmin)
    print(f"\nTest dogrulugu: %{dogruluk * 100:.1f}")
    print("\nHarf bazinda detay:")
    print(classification_report(y_test, tahmin))

    with open(MODEL_DOSYASI, "wb") as f:
        pickle.dump(model, f)
    print(f"Model kaydedildi -> {MODEL_DOSYASI}")

    if dogruluk < 0.9:
        print("\nIpucu: Dogruluk dusukse her harf icin daha fazla ve daha")
        print("cesitli (farkli aci/mesafe/isik) ornek toplayin.")


if __name__ == "__main__":
    main()
