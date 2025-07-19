from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gdown
import os

# === 1. Télécharger le modèle et le dataset si besoin ===

# Lien Google Drive de ton dataset (remplace par le tien si besoin)
dataset_url = 'https://drive.google.com/uc?id=1axkdHBVwq8zBKrBROArOeRA4eTXzmCxK'
dataset_path = 'fakealert_dataset_complet.csv'

if not os.path.exists(dataset_path):
    print("Téléchargement du dataset depuis Google Drive...")
    gdown.download(dataset_url, dataset_path, quiet=False)

# === 2. Charger le modèle BERT ===
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# === 3. Charger le dataset ===
df = pd.read_csv(dataset_path)

# === 4. Encoder les phrases du dataset ===
corpus_embeddings = model.encode(df["information"].tolist(), convert_to_tensor=True)

# === 5. Initialiser l'application Flask ===
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return analyze()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    input_text = data.get("text")

    if not input_text:
        return jsonify({"error": "Aucun texte fourni"}), 400

    # Encoder la phrase de l'utilisateur
    input_embedding = model.encode(input_text, convert_to_tensor=True)

    # Calcul des similarités
    similarities = util.cos_sim(input_embedding, corpus_embeddings)[0]
    best_match_index = int(similarities.argmax())
    best_score = round(float(similarities[best_match_index]), 2)

    if best_score < 0.4:
        return jsonify({
            "prediction": "Inconnu",
            "score": best_score,
            "explication": "Aucune correspondance suffisante trouvée dans notre base de données.",
            "source": "N/A"
        })

    result = {
        "prediction": "Real News" if df.loc[best_match_index, "label"] == 1 else "Fake News",
        "score": best_score,
        "explication": df.loc[best_match_index, "explication"],
        "source": df.loc[best_match_index, "source"]
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)