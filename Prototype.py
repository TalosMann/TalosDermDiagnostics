import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from fastai.learner import load_learner

# Load your pre-trained model (ensure the model is in the same directory as your app.py or provide the full path)
model_path = "models/body_images_resnet50.pkl"
model = load_learner(model_path)  # Correct way to load FastAI exported models

# Streamlit app layout
st.title("Skin Disease Classification App")
st.write("Upload an image to get the top 3 possible skin conditions.")

# Image upload functionality
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction using FastAI's predict method
    pred_class, pred_idx, outputs = model.predict(image)

    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(outputs, 3)  # Get top 3 values and indices
    top3_probs = top3_probs.tolist()  # Convert to list for display
    top3_indices = top3_indices.tolist()  # Convert to list

    # Display top 3 predictions with confidence scores
    st.write("### Top 3 Predictions:")
    for i in range(3):
        class_name = model.dls.vocab[top3_indices[i]]  # Get class label
        probability = top3_probs[i] * 100  # Convert to percentage
        st.write(f"**{class_name}** - {probability:.2f}% confidence")
