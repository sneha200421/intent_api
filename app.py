from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib

# Initialize Flask app
app = Flask(__name__)

# === Load model & tokenizer directly from Hugging Face ===
MODEL_NAME = "sneha21052004/intent_model"  # 🔹 Replace with your Hugging Face repo name

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Load your label encoder (since it’s not part of the model)
label_encoder = joblib.load("label_encoder.pkl")

# === Define predict route ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_label_id = torch.argmax(logits, dim=1).item()
    pred_label = label_encoder.inverse_transform([pred_label_id])[0]

    return jsonify({
        "text": text,
        "predicted_label": pred_label
    })

# === Health check route ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Intent Classification API is running ✅"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
