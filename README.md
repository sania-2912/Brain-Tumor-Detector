# Brain-Tumor-Detector
It is a trained Machine Learning model that will detect the type of Tumor based on the image uploaded and will give you the symptoms of the type


A deep learning-powered web application that detects brain tumor types from MRI images. This project uses a custom-trained convolutional neural network (CNN) to classify the uploaded image into one of four categories: glioma, meningioma, pituitary, or no tumor. It also displays common symptoms related to the detected tumor type to enhance user understanding.

🌟 Demo
Upload an MRI scan and the model will:
Predict if a tumor is present.
Identify the tumor type.
Display relevant symptoms.
It aims to serve as a proof-of-concept for early medical diagnostic assistance.

📄 Dataset Used
The model is trained using the publicly available dataset from Kaggle:
Brain Tumor Classification (MRI)https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

It contains 4 classes:
Glioma Tumor
Meningioma Tumor
Pituitary Tumor
No Tumor

🤖 Model Training
The model is a custom CNN built using PyTorch, trained from scratch with the following specs:
Input Size: 64x64 grayscale images
Architecture: 3 convolutional layers + 2 fully connected layers
Optimizer: Adam
Loss Function: Cross Entropy Loss
Accuracy Achieved: ~94% on validation set

Preprocessing Steps:
Grayscale conversion
Image resizing to 64x64
Normalization
The trained model is saved as brain_tumor_model.pth and used during inference.

🌐 Live Web Application
The app is built using Flask for the backend and basic HTML/CSS for the frontend with a neural-style animated brain background. It allows users to upload images and view predictions.
Features:
Displays uploaded MRI image
Shows tumor type and related symptoms

🚀 How to Run Locally
Clone the repository:
git clone https://github.com/yourusername/Brain-Tumor-Detector
cd Brain-Tumor-Detector

Create a virtual environment:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

Install dependencies:
pip install -r requirements.txt

Run the app:
python app/app.py

Open http://127.0.0.1:5000 in your browser.

Thought behind the project:
Brain tumors are life-threatening and require early diagnosis. This project shows how AI can assist in medical diagnosis and make initial screening faster and more accessible.
Note: This is not a substitute for professional medical advice or diagnosis.

🙏 Acknowledgements
Kaggle Dataset
PyTorch, Flask, PIL, torchvision
All contributors and testers

💼 License
MIT License - free for personal and educational use.
