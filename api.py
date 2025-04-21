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

# Class labels (you may need to update this depending on your model output)
class_labels = {
    0: "Anxiety",
    1: "Depression",
    2: "PTSD",
    3: "Stress-Related Disorder",
    4: "Substance Use Disorder",
    5: "Eating Disorder",
    6: "Self-Harm Challenges",
    7: "Attention Issues"
}

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

        # Get the predicted class and the confidence score
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        predicted_condition = class_labels.get(predicted_class_id, "Unknown Condition")

        return jsonify({
            "predicted_condition": predicted_condition,
            "confidence_score": round(torch.softmax(logits, dim=-1)[0][predicted_class_id].item() * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For platforms like Render (port 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
