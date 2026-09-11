# AgriVision

AgriVision is an AI-powered plant-disease detection project built with Flask and PyTorch. The application allows users to upload an image of a plant leaf and analyzes it to identify possible diseases across 39 plant-health categories.

## Project purpose

Plant diseases can reduce crop quality and productivity when they are not identified early. AgriVision demonstrates how artificial intelligence and image classification can help users recognize plant diseases and access useful information from a simple web interface.

## Features

- Upload a plant-leaf image for disease classification
- Identify plant conditions across 39 categories
- Display the predicted disease name and description
- Provide possible prevention and treatment guidance
- Recommend relevant agricultural supplements
- Browse available supplements through the Agro Medicines section
- Use a responsive Flask-based web interface

## Technologies used

- Python
- Flask
- PyTorch and TorchVision
- Convolutional Neural Network (CNN)
- Pandas and NumPy
- Pillow
- HTML, CSS, Bootstrap, and Jinja templates
- Jupyter Notebook

## Repository contents

- `ML Model/` — Jupyter Notebook containing the image-processing, CNN, training, and evaluation workflow
- `Website/` — Flask application, CNN architecture, HTML templates, and disease and supplement data

## How it works

1. A user uploads an image of a plant leaf.
2. The image is resized and converted into a tensor.
3. The CNN analyzes the image and predicts its plant-disease category.
4. The application retrieves the corresponding disease information.
5. The results page presents the disease description, prevention guidance, and suggested agricultural supplement.

## Project scope

AgriVision was developed as an academic artificial-intelligence project to demonstrate the practical use of deep learning, image classification, and web development in agriculture.
