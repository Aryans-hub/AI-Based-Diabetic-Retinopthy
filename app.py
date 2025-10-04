import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import tensorflow as tf

# Load the pre-trained model
@st.cache_resource
def load_dr_model():
    model = load_model('dr_detection_model.h5')
    return model

model = load_dr_model()

# Define the classes
class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

def get_grad_cam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def superimpose_heatmap(img, heatmap):
    # Convert PIL Image to a NumPy array for OpenCV
    img_np = np.array(img.convert('RGB'))

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    # Convert heatmap to a color map (e.g., JET)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    # The addWeighted function blends two images
    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    # Return the result
    return superimposed_img

st.title("AI-Based Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image for diagnosis.")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image for the model
    img_array = img_to_array(image.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = np.argmax(score)
    
    st.write(f"Prediction: **{class_names[predicted_class]}**")
    st.write(f"Confidence: {100 * np.max(score):.2f}%")

    # Generate and display Grad-CAM heatmap
    st.subheader("Model Interpretability")
    st.write("Below is a heatmap showing the regions of interest the model focused on.")
    
    last_conv_layer_name = "conv5_block3_out"

    try:
        # Use the correctly defined layer name in the function call
        heatmap = get_grad_cam_heatmap(img_array, model, last_conv_layer_name)
        superimposed_img = superimpose_heatmap(image, heatmap)
        st.image(superimposed_img, caption='Grad-CAM Heatmap', use_container_width=True)
    except Exception as e:
        st.write(f"Could not generate Grad-CAM visualization: {e}")