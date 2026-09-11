# AgriVision

AgriVision is a Flask and PyTorch plant-disease detection project. It accepts a plant-leaf image, predicts one of 39 classes, displays disease information and prevention guidance, and lists relevant agricultural supplements.

## Repository contents

- `ML Model/` — model-training notebook
- `Website/` — Flask application, CNN architecture, HTML templates, and CSV data

## Model file

The trained weights are intentionally not included yet. Prediction requires:

```text
Website/plant_disease_model_1_latest.pt
```

The website can start without that file, but the prediction endpoint returns a clear unavailable message until the weights are added.

## Local setup

```bash
cd Website
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Training

Use `ML Model/Plant Disease Detection Code.ipynb` with the original 39-class image dataset. After training:

```python
torch.save(model.state_dict(), "plant_disease_model_1_latest.pt")
```

Move the resulting file into `Website/`.
