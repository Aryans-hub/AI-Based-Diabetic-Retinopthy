import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the dataset CSV file
# You would need to download the APTOS 2019 dataset from Kaggle
# and place the train.csv file and images in the appropriate folders.
train_df = pd.read_csv(r'A:\Aryan Drive D\Projects\Diabetic Retinopathy\Dataset\train.csv')
train_path = r'A:\Aryan Drive D\Projects\Diabetic Retinopathy\Dataset\train_images'

# Function to preprocess images (resize and normalize)
def preprocess_image(image_path, size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    # You can add more advanced preprocessing techniques here,
    # like Ben's preprocessing or circular cropping.
    return img

# Prepare the data
images = []
labels = []
for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
    image_id = row['id_code']
    label = row['diagnosis']
    image_path = os.path.join(train_path, f'{image_id}.png')
    
    if os.path.exists(image_path):
        preprocessed_img = preprocess_image(image_path)
        images.append(preprocessed_img)
        labels.append(label)

X = np.array(images)
y = np.array(labels)