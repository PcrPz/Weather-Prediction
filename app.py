# app.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import time
import io
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download

# Configuration
UPLOAD_FOLDER = '/Users/ftmacbookair/Desktop/untitled folder/photo'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels for the model
LABELS = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']

# Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    try:
        # Create a ResNet50 model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        
        # Download the model weights directly from Hugging Face
        model_path = hf_hub_download(repo_id="PcrPz/Weather_Resnet50", filename="best_model.pth")
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle potential 'module.' prefix if the model was trained with DataParallel
        if all(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        
        model.eval()
        
        print(f"Model loaded successfully from Hugging Face Hub")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        print(f"Error loading model: {str(e)}")
        return None

def predict_image(image_data, is_path=False):
    try:
        # Load model 
        model = load_model()
        
        if model is None:
            return [{"class": "Error", "probability": 100.0, "error": "Failed to load model"}]
        
        # Image transformation - standard for ResNet
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Process image
        if is_path:
            if isinstance(image_data, str):
                img = Image.open(image_data).convert('RGB')
            else:
                img = image_data.convert('RGB')
        else:
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        img_tensor = transform(img).unsqueeze(0)
        
        # Perform inference
        start_time = time.time()
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, min(5, len(LABELS)))
        
        results = []
        for i in range(top_indices.size(0)):
            class_idx = top_indices[i].item()
            results.append({
                'class': LABELS[class_idx],
                'probability': float(top_probs[i].item()) * 100,
                'inference_time': inference_time
            })
        
        return results
    except Exception as e:
        print(f"Prediction error: {e}")
        st.error(f"Prediction error: {e}")
        return [{"class": "Error", "probability": 100.0, "error": str(e)}]

# Set page config
st.set_page_config(
    page_title="Weather Image Classifier",
    page_icon="🌦️",
    layout="wide"
)

# App title and description
st.title("Weather Image Classifier")
st.markdown("Upload an image to classify the weather condition (cloudy, foggy, rainy, shine, sunrise)")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# When an image is uploaded
if uploaded_file is not None:
    # Display the image
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction on button click
    with col2:
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                # Get image bytes
                img_bytes = uploaded_file.getvalue()
                
                # Save the file
                filename = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(filename, "wb") as f:
                    f.write(img_bytes)
                
                # Predict
                predictions = predict_image(img_bytes)
                
                # Display results
                st.subheader("Classification Results:")
                
                if 'error' in predictions[0]:
                    st.error(predictions[0]['error'])
                else:
                    # Create a DataFrame for better display
                    results_df = pd.DataFrame(predictions)
                    
                    # Display the inference time
                    if predictions and 'inference_time' in predictions[0]:
                        st.info(f"Inference time: {predictions[0]['inference_time']:.4f} seconds")
                    
                    # Remove inference_time column for display
                    if 'inference_time' in results_df.columns:
                        results_df = results_df.drop(columns=['inference_time'])
                    
                    # Rename columns for better display
                    results_df = results_df.rename(columns={
                        'class': 'Weather Type',
                        'probability': 'Confidence (%)'
                    })
                    
                    # Format probability as percentage with 2 decimal places
                    results_df['Confidence (%)'] = results_df['Confidence (%)'].apply(lambda x: f"{x:.2f}%")
                    
                    # Display the results
                    st.table(results_df)
                    
                    # Display the top prediction prominently
                    if predictions:
                        st.success(f"Top prediction: **{predictions[0]['class']}** with {predictions[0]['probability']:.2f}% confidence")

# Add a test image option with URL
st.sidebar.header("Test with Sample Image")
sample_url = st.sidebar.text_input("Enter image URL or use default", 
                                  value="https://images.pexels.com/photos/209831/pexels-photo-209831.jpeg")

if st.sidebar.button("Test with Image URL"):
    try:
        with st.sidebar.spinner("Downloading image..."):
            response = requests.get(sample_url)
            image = Image.open(BytesIO(response.content))
            st.sidebar.image(image, caption="Test Image", use_column_width=True)
        
        with st.sidebar.spinner("Classifying..."):
            test_results = predict_image(image, is_path=True)
            
            # Display test results
            st.sidebar.subheader("Test Results:")
            if 'error' in test_results[0]:
                st.sidebar.error(test_results[0]['error'])
            else:
                for result in test_results:
                    st.sidebar.write(f"- {result['class']}: {result['probability']:.2f}%")
                if test_results and 'inference_time' in test_results[0]:
                    st.sidebar.info(f"Inference time: {test_results[0]['inference_time']:.4f} seconds")
    except Exception as e:
        st.sidebar.error(f"Error processing image: {e}")

# Add information about the model
st.sidebar.header("About")
st.sidebar.info("""
This application uses a ResNet50 model from Hugging Face (PcrPz/Weather_Resnet50) 
to classify weather conditions in images.
The model can identify weather types: cloudy, foggy, rainy, shine, and sunrise.
""")

# Model details
st.sidebar.header("Model Details")
st.sidebar.write("Architecture: ResNet50")
st.sidebar.write("Source: Hugging Face (PcrPz/Weather_Resnet50)")
st.sidebar.write(f"Classes: {', '.join(LABELS)}")