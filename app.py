"""
Darija Sentiment — YouTube Analyser
Backend Flask qui :
  1. Récupère les commentaires d'une vidéo YouTube (API YouTube Data v3)
  2. Les classe (positif / négatif / neutre) avec le modèle SVM entraîné
  3. Renvoie les statistiques attendues par index.html

Avant de lancer cette app, entraîne le modèle une fois :
    python train_model.py

Configuration : définis ta clé API YouTube dans la variable d'environnement
YOUTUBE_API_KEY (ou directement ci-dessous, à éviter en production / GitHub
public).

Lancement :
    pip install -r requirements.txt
    export YOUTUBE_API_KEY="ta_clé_ici"
    python app.py
"""

import os
import re
import joblib
from flask import Flask, request, jsonify, render_template
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ── Configuration ──────────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
MODEL_PATH = "model.pkl"

app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")

# ── Chargement du modèle entraîné ───────────────────────────────────────────
_model_bundle = None


def get_model_bundle():
    """Charge le modèle une seule fois (lazy loading)."""
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"{MODEL_PATH} introuvable. Lance d'abord : python train_model.py"
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


# ── Extraction de l'ID vidéo depuis une URL YouTube ────────────────────────
def extract_video_id(url):
    """Extrait l'ID vidéo de différents formats d'URL YouTube (ou un ID brut)."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Si l'utilisateur a collé directement un ID vidéo brut (11 caractères)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url.strip()):
        return url.strip()
    return None


# ── Récupération des infos vidéo + commentaires via l'API YouTube ──────────
def fetch_video_info(youtube, video_id):
    resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    items = resp.get("items", [])
    if not items:
        return None
    snippet = items[0]["snippet"]
    stats = items[0].get("statistics", {})
    return {
        "title": snippet.get("title", "Vidéo YouTube"),
        "channel": snippet.get("channelTitle", "—"),
        "thumb": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
        "views": int(stats.get("viewCount", 0)),
    }


def fetch_comments(youtube, video_id, max_comments):
    comments = []
    page_token = None

    while len(comments) < max_comments:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=page_token,
            order="relevance",
            textFormat="plainText",
        ).execute()

        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": top.get("textDisplay", ""),
                "author": top.get("authorDisplayName", "Anonyme"),
                "likes": top.get("likeCount", 0),
            })
            if len(comments) >= max_comments:
                break

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return comments


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model_info")
def model_info():
    try:
        bundle = get_model_bundle()
        return jsonify({
            "name": bundle.get("model_name", "SVM (LinearSVC)"),
            "accuracy": bundle.get("accuracy"),
            "f1_macro": bundle.get("f1_macro"),
        })
    except FileNotFoundError as e:
        return jsonify({"name": "Modèle non chargé", "error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    max_comments = int(data.get("max_comments") or 100)
    max_comments = max(10, min(max_comments, 1000))

    if not url:
        return jsonify({"error": "Merci de fournir un lien YouTube."}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Lien YouTube invalide."}), 400

    if not YOUTUBE_API_KEY:
        return jsonify({
            "error": "Clé API YouTube manquante. Définis la variable "
                     "d'environnement YOUTUBE_API_KEY."
        }), 500

    try:
        bundle = get_model_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        info = fetch_video_info(youtube, video_id)
        if info is None:
            return jsonify({"error": "Vidéo introuvable."}), 404

        raw_comments = fetch_comments(youtube, video_id, max_comments)
    except HttpError as e:
        return jsonify({"error": f"Erreur API YouTube : {e}"}), 502

    if not raw_comments:
        return jsonify({"error": "Aucun commentaire trouvé pour cette vidéo."}), 404

    # ── Classification des commentaires ─────────────────────────────────
    pipeline = bundle["pipeline"]
    texts = [c["text"] for c in raw_comments]
    predictions = pipeline.predict(texts)

    for c, label in zip(raw_comments, predictions):
        c["sentiment"] = label

    total = len(raw_comments)
    counts = {
        "positif": sum(1 for c in raw_comments if c["sentiment"] == "positif"),
        "negatif": sum(1 for c in raw_comments if c["sentiment"] == "négatif"),
        "neutre": sum(1 for c in raw_comments if c["sentiment"] == "neutre"),
    }
    pct = {k: round(v / total * 100) for k, v in counts.items()}

    # Timeline : séquence ordonnée des labels (utilisée par le front pour
    # les graphiques par tranches et la moyenne glissante). On mappe le
    # label 'négatif' -> 'negatif' (sans accent) pour matcher le JS du front.
    label_map = {"positif": "positif", "négatif": "negatif", "neutre": "neutre"}
    timeline = [label_map.get(c["sentiment"], c["sentiment"]) for c in raw_comments]

    def top_n(label, n=10):
        filtered = [c for c in raw_comments if c["sentiment"] == label]
        filtered.sort(key=lambda c: c["likes"], reverse=True)
        return [
            {"text": c["text"], "author": c["author"], "likes": c["likes"]}
            for c in filtered[:n]
        ]

    response = {
        "info": info,
        "total": total,
        "counts": counts,
        "pct": pct,
        "timeline": timeline,
        "top_pos": top_n("positif"),
        "top_neg": top_n("négatif"),
        "top_neu": top_n("neutre"),
    }
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
