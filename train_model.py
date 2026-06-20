"""
Entraîne le pipeline TF-IDF + SVM (meilleur modèle du notebook) sur
dataset_pretraite.csv et le sauvegarde dans model.pkl pour app.py.

Usage : python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH = "dataset_pretraite.csv"
MODEL_PATH = "model.pkl"


def main():
    print(f"Chargement de {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("")
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    print(f"Commentaires d'entraînement : {len(df)}")
    print(df["sentiment"].value_counts())

    X = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline identique à celui du notebook (meilleur modèle : SVM)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=2
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])

    print("\nEntraînement du modèle SVM (LinearSVC)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nAccuracy  : {acc*100:.1f}%")
    print(f"F1 macro  : {f1*100:.1f}%")
    print("\nRapport détaillé :")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Ré-entraîner sur 100% des données avant de sauvegarder (meilleure
    # utilisation des données pour le modèle final servi en production)
    print("\nRé-entraînement sur l'ensemble complet du dataset...")
    pipeline.fit(X, y)

    joblib.dump({
        "pipeline": pipeline,
        "labels": sorted(y.unique().tolist()),
        "model_name": "SVM (LinearSVC)",
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
    }, MODEL_PATH)

    print(f"\n✅ Modèle sauvegardé sous {MODEL_PATH}")


if __name__ == "__main__":
    main()
