import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

import CNN


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "plant_disease_model_1_latest.pt"

disease_info = pd.read_csv(BASE_DIR / "disease_info.csv", encoding="cp1252")
supplement_info = pd.read_csv(BASE_DIR / "supplement_info.csv", encoding="cp1252")

app = Flask(__name__, template_folder="HTML Templates")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

model = None
if MODEL_PATH.exists():
    model = CNN.CNN(39)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()


def prediction(image_path):
    if model is None:
        raise RuntimeError(
            "The trained model file plant_disease_model_1_latest.pt is missing."
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((224, 224))
        input_data = TF.to_tensor(image).view((-1, 3, 224, 224))

    with torch.no_grad():
        output = model(input_data)

    return int(np.argmax(output.numpy()))


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/index")
def ai_engine_page():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    if model is None:
        return (
            "Prediction is unavailable because plant_disease_model_1_latest.pt "
            "has not been added.",
            503,
        )

    image = request.files.get("image")
    if image is None or not image.filename:
        return "Please select an image.", 400

    filename = secure_filename(image.filename)
    file_path = UPLOAD_DIR / filename
    image.save(file_path)

    pred = prediction(file_path)
    return render_template(
        "submit.html",
        title=disease_info["disease_name"][pred],
        desc=disease_info["description"][pred],
        prevent=disease_info["Possible Steps"][pred],
        image_url=disease_info["image_url"][pred],
        pred=pred,
        sname=supplement_info["supplement name"][pred],
        simage=supplement_info["supplement image"][pred],
        buy_link=supplement_info["buy link"][pred],
    )


@app.route("/market", methods=["GET", "POST"])
def market():
    return render_template(
        "market.html",
        supplement_image=list(supplement_info["supplement image"]),
        supplement_name=list(supplement_info["supplement name"]),
        disease=list(disease_info["disease_name"]),
        buy=list(supplement_info["buy link"]),
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
