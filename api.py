from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Define the model directory
model_name = "vincentbaldon2003/mental-health-distilbert"

# Load tokenizer and model from Hugging Face Model Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Class labels (directly from your dataset)
prediction_condition_labels = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Gibberish",
    "Bipolar",
    "Stress",
    "Personality disorder"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        responses = data.get("responses", [])

        # Ensure there are at least 5 responses
        if not responses or len(responses) < 5:
            return jsonify({"error": "Invalid input. Expected at least 5 responses."}), 400

        # Combine responses into one text input
        combined_text = " ".join(responses)

        # Tokenize the combined text
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the top 3 predicted classes and their confidence scores
        top_k = 3
        probs = torch.softmax(logits, dim=-1)
        top_k_values, top_k_indices = torch.topk(probs, top_k, dim=-1)

        # Get the top 3 predicted conditions and their confidence scores
        top_3_predictions = [
            {
                "predicted_condition": prediction_condition_labels[idx.item()],
                "confidence_score": round(prob.item() * 100, 2)
            }
            for idx, prob in zip(top_k_indices[0], top_k_values[0])
        ]

        return jsonify({
            "top_3_predictions": top_3_predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For platforms like Render (port 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
