import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- START: Data Preparation Code ---
# Load the dataset CSV file
train_df = pd.read_csv('A:/Aryan Drive D/Projects/Diabetic Retinopathy/Dataset/train.csv')
train_path = 'A:/Aryan Drive D/Projects/Diabetic Retinopathy/Dataset/train_images'

# Function to preprocess images
def preprocess_image(image_path, size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
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
# --- END: Data Preparation Code ---

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add new custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, y_train, 
          epochs=10, 
          batch_size=32, 
          validation_data=(X_val, y_val))

# Save the trained model
model.save('dr_detection_model.h5')