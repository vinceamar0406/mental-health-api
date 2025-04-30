from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Model and tokenizer setup
model_name = "vincentbaldon2003/mental-health-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Class labels matching your dataset
status_labels = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Bipolar",
    "Stress",
    "Personality disorder"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        responses = data.get("responses", [])

        # Validate that at least 5 responses exist
        if not responses or len(responses) < 5:
            return jsonify({"error": "Invalid input. Expected at least 5 responses."}), 400

        # Merge responses into a single string
        combined_text = " ".join(responses)

        # Tokenize
        inputs = tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the top (most confident) prediction
        probs = torch.softmax(logits, dim=-1)
        top_pred_index = torch.argmax(probs, dim=-1).item()
        predicted_condition = status_labels[top_pred_index]
        confidence_score = round(probs[0][top_pred_index].item() * 100, 2)

        # Return only the top prediction in the format Laravel expects
        return jsonify({
            "predicted_condition": predicted_condition,
            "confidence_score": confidence_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
