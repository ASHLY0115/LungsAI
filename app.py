import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io

app = Flask(__name__)
CORS(app)

# Load Model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "efficientnet_focal_v1.pt")
device = torch.device("cpu")

print(f"Loading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=device)

model = timm.create_model(
    checkpoint["config"]["model"]["backbone"],
    pretrained=False,
    num_classes=checkpoint["config"]["data"]["num_classes"],
    drop_rate=checkpoint["config"]["model"]["dropout"]
)

model.load_state_dict(checkpoint["model_state"])
model.eval()
print("Model loaded successfully.")
print(f"Model Config: {checkpoint['config']}")

classes = [
    "Adenocarcinoma",
    "Squamous Cell Carcinoma",
    "Large Cell Carcinoma",
    "Normal"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

GRAYSCALE_CHANNEL_DELTA_THRESHOLD = 8
MAX_COLOR_PIXEL_RATIO = 0.05
MAX_MEAN_SATURATION = 8
MAX_SATURATION = 20

CT_VALIDATION_BOUNDS = {
    "mean_min": 0.15,
    "mean_max": 0.60,
    "std_min": 0.10,
    "body_ratio_min": 0.45,
    "body_ratio_max": 1.0,
    "dark_ratio_min": 0.05,
    "left_dark_min": 0.04,
    "right_dark_min": 0.04,
}


def analyze_image_domain(image):
    rgb_array = np.asarray(image.convert("RGB"), dtype=np.int16)
    channel_delta = rgb_array.max(axis=2) - rgb_array.min(axis=2)
    color_pixel_ratio = float((channel_delta > GRAYSCALE_CHANNEL_DELTA_THRESHOLD).mean())

    hsv_array = np.asarray(image.convert("HSV"), dtype=np.float32)
    saturation = hsv_array[:, :, 1]
    mean_saturation = float(saturation.mean())
    max_saturation = float(saturation.max())

    grayscale = image.convert("L").resize((224, 224))
    grayscale_array = np.asarray(grayscale, dtype=np.float32) / 255.0
    center = grayscale_array[22:202, 22:202]
    body_mask = center > 0.12

    if not body_mask.any():
        return {
            "is_valid": False,
            "reason": "The uploaded image does not resemble a chest CT scan.",
        }

    left_half = center[:, :90]
    right_half = center[:, 90:]
    left_body_mask = left_half > 0.12
    right_body_mask = right_half > 0.12

    body_pixels = int(body_mask.sum())
    left_body_pixels = max(int(left_body_mask.sum()), 1)
    right_body_pixels = max(int(right_body_mask.sum()), 1)

    metrics = {
        "mean": float(center.mean()),
        "std": float(center.std()),
        "body_ratio": float(body_mask.mean()),
        "dark_ratio": float(((center < 0.28) & body_mask).sum() / body_pixels),
        "left_dark_ratio": float(((left_half < 0.28) & left_body_mask).sum() / left_body_pixels),
        "right_dark_ratio": float(((right_half < 0.28) & right_body_mask).sum() / right_body_pixels),
        "color_pixel_ratio": color_pixel_ratio,
        "mean_saturation": mean_saturation,
        "max_saturation": max_saturation,
    }

    is_grayscale_like = (
        metrics["color_pixel_ratio"] <= MAX_COLOR_PIXEL_RATIO
        and metrics["mean_saturation"] <= MAX_MEAN_SATURATION
        and metrics["max_saturation"] <= MAX_SATURATION
    )

    looks_like_ct = (
        CT_VALIDATION_BOUNDS["mean_min"] <= metrics["mean"] <= CT_VALIDATION_BOUNDS["mean_max"]
        and metrics["std"] >= CT_VALIDATION_BOUNDS["std_min"]
        and CT_VALIDATION_BOUNDS["body_ratio_min"] <= metrics["body_ratio"] <= CT_VALIDATION_BOUNDS["body_ratio_max"]
        and metrics["dark_ratio"] >= CT_VALIDATION_BOUNDS["dark_ratio_min"]
        and metrics["left_dark_ratio"] >= CT_VALIDATION_BOUNDS["left_dark_min"]
        and metrics["right_dark_ratio"] >= CT_VALIDATION_BOUNDS["right_dark_min"]
    )

    if not is_grayscale_like:
        return {
            "is_valid": False,
            "reason": "The uploaded image appears to be a color photograph. Please upload a grayscale chest CT scan.",
            "metrics": metrics,
        }

    if not looks_like_ct:
        return {
            "is_valid": False,
            "reason": "The uploaded image does not resemble a chest CT scan. Please upload a valid lung CT slice for analysis.",
            "metrics": metrics,
        }

    return {
        "is_valid": True,
        "metrics": metrics,
    }

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    validation = analyze_image_domain(image)
    if not validation["is_valid"]:
        return jsonify({
            "error": validation["reason"],
            "predictionResult": "Invalid Image Format",
            "confidenceScore": 0,
            "riskLevel": "Low",
            "status": "error"
        }), 400

    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()
        
        # Debug Logging
        print(f"Raw Outputs: {outputs}")
        print(f"Probabilities: {probabilities}")
        print(f"Predicted Class Index: {predicted_class_idx} ({classes[predicted_class_idx]})")
    
    prediction = classes[predicted_class_idx]
    
    # Detailed probabilities for all classes
    all_probabilities = {classes[i]: round(probabilities[i].item() * 100, 2) for i in range(len(classes))}
    
    # Determine risk level and recommendation
    risk_level = "Low"
    recommendation = "No immediate action required. Maintain regular check-ups."
    
    if prediction != "Normal":
        if confidence > 0.8:
            risk_level = "High"
            recommendation = "URGENT: Highly suspicious of malignancy. Immediate biopsy or surgical consultation recommended."
        else:
            risk_level = "Medium"
            recommendation = "Suspicious findings. Follow-up CT scan in 3 months or further diagnostic tests recommended."
    elif confidence < 0.6:
        risk_level = "Medium"
        recommendation = "Inconclusive results. Repeat scan or additional imaging recommended for clarity."

    return jsonify({
        "predictionResult": prediction,
        "confidenceScore": round(confidence * 100, 2),
        "riskLevel": risk_level,
        "recommendation": recommendation,
        "probabilities": all_probabilities,
        "status": "success"
    })

if __name__ == "__main__":
    app.run(port=5000)
