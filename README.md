README.md Content
AI-Based Detection of Diabetic Retinopathy Using Retinal Fundus Images 🩺
This project develops an explainable AI model for the automated detection of diabetic retinopathy (DR), a leading cause of blindness. The model processes retinal fundus images and classifies them into one of five severity classes, providing a fast and accessible screening tool.

🌟 Features

Advanced AI Model: Uses transfer learning with a pre-trained ResNet50 model, which is highly effective for complex image patterns found in medical images. The model achieves a robust performance with a 

92% F1-score.


Model Explainability: Implements Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the specific regions of interest the AI model focuses on to make its diagnosis. This helps build trust and provides insights into the model's decision-making process.


User-Friendly Web App: Deploys a web application using Streamlit for real-time predictions. Users can upload an image and receive an instant diagnosis along with the Grad-CAM visualization.


Scalable Solution: The project is inspired by initiatives like CDAC Mohali's SightSaver and is designed as a low-cost, scalable AI solution to support early screening in low-resource settings.

🛠️ Technology Stack
Deep Learning Framework: TensorFlow, Keras


Model Architectures: ResNet50 (primary), EfficientNet (for comparison) 

Data Handling: Pandas, NumPy

Image Processing: OpenCV, PIL

Web Framework: Streamlit

Libraries: scikit-learn, tqdm

🚀 Getting Started
Prerequisites
Python 3.8+

Git

Conda or a virtual environment (recommended)

Installation
Clone the Repository:

Bash

git clone https://github.com/your-username/your-repository.git
cd your-repository
Create and Activate Environment:

Bash

conda create -n retinopathy-env python=3.9
conda activate retinopathy-env
Install Dependencies:

Bash

pip install -r requirements.txt
(Note: You'll need to create a requirements.txt file listing all the libraries.)

📂 Dataset
The model was trained on the 

APTOS 2019 Blindness Detection dataset from Kaggle, which includes five classes of labeled retinal fundus images.

🧠 Model Architecture
The project uses 

transfer learning with a pre-trained ResNet50 model. ResNet50 is a deep neural network with 50 layers that uses "skip connections" to overcome the vanishing gradient problem in very deep networks, making it both stable and accurate.




The model was fine-tuned on the medical images and trained to classify them into the following categories:

0: No DR

1: Mild DR

2: Moderate DR

3: Severe DR

4: Proliferative DR

💡 Model Interpretability (Grad-CAM)
The project utilizes Grad-CAM, which generates a visual heatmap highlighting the pixels in the image that most influenced the model's prediction. This provides a transparent and "explainable" diagnosis.

🌐 Web App Deployment
The Streamlit web app allows users to upload an image and get an instant prediction with a confidence score and a Grad-CAM visualization. This makes the tool accessible and intuitive for non-technical users.

To run the app locally, execute the following command in your terminal:

Bash

streamlit run app.py
This will launch the app in your web browser.