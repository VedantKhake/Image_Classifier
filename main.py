# Import necessary libraries
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

# Function to load the pretrained MobileNetV2 model
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

# Function to preprocess the uploaded image to the format MobileNetV2 expects
def prepare_image(pil_image):
    np_image = np.array(pil_image)  # Convert to NumPy array
    resized_image = cv2.resize(np_image, (224, 224))  # Resize to 224x224
    preprocessed = preprocess_input(resized_image)  # Apply model-specific preprocessing
    return np.expand_dims(preprocessed, axis=0)  # Add batch dimension

# Function to classify image and return top 3 predictions
def get_predictions(model, image):
    try:
        processed = prepare_image(image)
        preds = model.predict(processed)
        top_preds = decode_predictions(preds, top=3)[0]  # Get top 3 predictions
        return top_preds
    except Exception as err:
        st.error(f"Something went wrong while classifying the image: {str(err)}")
        return None

# Main Streamlit app logic
def main():
    st.set_page_config(page_title="Animal Image Classifier", page_icon="🐾", layout="centered")

    st.title("🐾 AI-Powered Animal Identifier")
    st.write("Upload a photo of an animal and let the AI guess what it is!")

    # Load the model only once using Streamlit cache
    @st.cache_resource
    def get_cached_model():
        return load_model()

    model = get_cached_model()

    # Allow user to upload image
    file = st.file_uploader("Upload an animal image (JPG or PNG)", type=["jpg", "png"])

    if file is not None:
        st.image(file, caption="Your Uploaded Image", use_container_width=True)

        if st.button("🧠 Identify Animal"):
            with st.spinner("Analyzing with AI..."):
                pil_img = Image.open(file)
                predictions = get_predictions(model, pil_img)

                if predictions:
                    st.subheader("🔍 AI Predictions")
                    highest_prob = 0.0
                    most_likely_label = ""

                    for _, label, prob in predictions:
                        st.write(f"**{label}** — {prob:.2%}")
                        if prob > highest_prob:
                            highest_prob = prob
                            most_likely_label = label

                    st.markdown(f"### ✅ The AI believes this is most likely a **{most_likely_label.replace('_', ' ').title()}**.")

# Run the app
if __name__ == "__main__":
    main()
