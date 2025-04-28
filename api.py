from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Model and tokenizer setup
model_name = "vincentbaldon2003/mental-health-distilbert-2"
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

        # Top 3 predictions
        probs = torch.softmax(logits, dim=-1)
        top_k_values, top_k_indices = torch.topk(probs, 3, dim=-1)

        # Format top 3 into a list of dictionaries
        top_3_predictions = [
            {
                "status": status_labels[idx.item()],
                "confidence_score": round(prob.item() * 100, 2)
            }
            for idx, prob in zip(top_k_indices[0], top_k_values[0])
        ]

        # Return the top 3 predictions
        return jsonify({
            "top_3_predictions": top_3_predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
