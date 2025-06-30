import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request
from src.model import BrainTumorCNN

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("app", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

SYMPTOMS = {
    'glioma': [
        'Seizures',
        'Morning headaches',
        'Personality changes',
        'Memory problems'
    ],
    'meningioma': [
        'Vision issues',
        'Hearing loss',
        'Limb weakness',
        'Difficulty concentrating'
    ],
    'pituitary': [
        'Hormonal imbalance',
        'Menstrual irregularities',
        'Vision disturbances',
        'Fatigue and weakness'
    ],
    'notumor': []
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    symptoms = []

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            image = Image.open(save_path).convert('L')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                prediction = CLASSES[predicted.item()]
                image_path = f"/static/uploads/{filename}"
                symptoms = SYMPTOMS[prediction]

    return render_template('index.html', prediction=prediction, image_path=image_path, symptoms=symptoms)

if __name__ == '__main__':
    app.run(debug=True)
