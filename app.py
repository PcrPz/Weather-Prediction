# app.py
import os
import tempfile
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

# Use a temporary directory for uploads instead of hardcoded path
UPLOAD_FOLDER = tempfile.gettempdir()

# Class labels for the models
LABELS = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']

# Model configurations
MODEL_CONFIGS = {
    "model1": {
        "name": "Weather ResNet50 A1",
        "repo_id": "PcrPz/Weather_Resnet50",
        "filename": "best_model_resnet50_a1.pth",
        "architecture": "resnet50",
        "num_classes": len(LABELS)
    },
    "model2": {
        "name": "Weather ResNet TV2",
        "repo_id": "PcrPz/Weather_Resnet50", 
        "filename": "best_model_resnet_tv2.pth", 
        "architecture": "resnet50",
        "num_classes": len(LABELS)
    }
}

# Load model from Hugging Face Hub
@st.cache_resource
def load_model(model_key):
    try:
        config = MODEL_CONFIGS[model_key]
        
        # Create appropriate model architecture
        if config["architecture"] == "resnet50":
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
        elif config["architecture"] == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, config["num_classes"])
        else:
            raise ValueError(f"Unsupported architecture: {config['architecture']}")
        
        # Download the model weights directly from Hugging Face
        model_path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle potential 'module.' prefix if the model was trained with DataParallel
        if all(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        
        model.eval()
        
        print(f"Model {config['name']} loaded successfully from Hugging Face Hub")
        return model
    except Exception as e:
        st.error(f"Error loading model {model_key}: {str(e)}")
        print(f"Error loading model {model_key}: {str(e)}")
        return None

def predict_image(image_data, model_key, is_path=False):
    try:
        # Load model 
        model = load_model(model_key)
        
        if model is None:
            return [{"class": "Error", "probability": 100.0, "error": f"Failed to load model {model_key}"}]
        
        # Image transformation - standard for most CNN models
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
        print(f"Prediction error with {model_key}: {e}")
        st.error(f"Prediction error with {model_key}: {e}")
        return [{"class": "Error", "probability": 100.0, "error": str(e)}]

# Set page config
st.set_page_config(
    page_title="Weather Image Classifier Comparison",
    page_icon="🌦️",
    layout="wide"
)

# App title and description
st.title("Weather Image Classifier - Model Comparison")
st.markdown("Upload an image to classify the weather condition and compare results between two models")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# When an image is uploaded
if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption="Uploaded Image", width=400)
    
    # Make prediction on button click
    if st.button("Classify with Both Models"):
        col1, col2 = st.columns(2)
        
        with st.spinner("Classifying..."):
            # Get image bytes and predict directly without saving
            img_bytes = uploaded_file.getvalue()
            
            # Get predictions from both models
            predictions1 = predict_image(img_bytes, "model1")
            predictions2 = predict_image(img_bytes, "model2")
            
            # Display results for model 1
            with col1:
                st.subheader(f"Model 1: {MODEL_CONFIGS['model1']['name']}")
                
                if 'error' in predictions1[0]:
                    st.error(predictions1[0]['error'])
                else:
                    # Create a DataFrame for better display
                    results_df1 = pd.DataFrame(predictions1)
                    
                    # Display the inference time
                    if predictions1 and 'inference_time' in predictions1[0]:
                        st.info(f"Inference time: {predictions1[0]['inference_time']:.4f} seconds")
                    
                    # Remove inference_time column for display
                    if 'inference_time' in results_df1.columns:
                        results_df1 = results_df1.drop(columns=['inference_time'])
                    
                    # Rename columns for better display
                    results_df1 = results_df1.rename(columns={
                        'class': 'Weather Type',
                        'probability': 'Confidence (%)'
                    })
                    
                    # Format probability as percentage with 2 decimal places
                    results_df1['Confidence (%)'] = results_df1['Confidence (%)'].apply(lambda x: f"{x:.2f}%")
                    
                    # Display the results
                    st.table(results_df1)
                    
                    # Display the top prediction prominently
                    if predictions1:
                        st.success(f"Top prediction: **{predictions1[0]['class']}** with {predictions1[0]['probability']:.2f}% confidence")
            
            # Display results for model 2
            with col2:
                st.subheader(f"Model 2: {MODEL_CONFIGS['model2']['name']}")
                
                if 'error' in predictions2[0]:
                    st.error(predictions2[0]['error'])
                else:
                    # Create a DataFrame for better display
                    results_df2 = pd.DataFrame(predictions2)
                    
                    # Display the inference time
                    if predictions2 and 'inference_time' in predictions2[0]:
                        st.info(f"Inference time: {predictions2[0]['inference_time']:.4f} seconds")
                    
                    # Remove inference_time column for display
                    if 'inference_time' in results_df2.columns:
                        results_df2 = results_df2.drop(columns=['inference_time'])
                    
                    # Rename columns for better display
                    results_df2 = results_df2.rename(columns={
                        'class': 'Weather Type',
                        'probability': 'Confidence (%)'
                    })
                    
                    # Format probability as percentage with 2 decimal places
                    results_df2['Confidence (%)'] = results_df2['Confidence (%)'].apply(lambda x: f"{x:.2f}%")
                    
                    # Display the results
                    st.table(results_df2)
                    
                    # Display the top prediction prominently
                    if predictions2:
                        st.success(f"Top prediction: **{predictions2[0]['class']}** with {predictions2[0]['probability']:.2f}% confidence")
            
            # Compare the results
            if not ('error' in predictions1[0]) and not ('error' in predictions2[0]):
                st.subheader("Model Comparison")
                
                # Create a comparison dataframe
                comparison_data = []
                for weather_type in LABELS:
                    prob1 = next((item['probability'] for item in predictions1 if item['class'] == weather_type), 0)
                    prob2 = next((item['probability'] for item in predictions2 if item['class'] == weather_type), 0)
                    comparison_data.append({
                        'Weather Type': weather_type,
                        f'{MODEL_CONFIGS["model1"]["name"]} (%)': f"{prob1:.2f}%",
                        f'{MODEL_CONFIGS["model2"]["name"]} (%)': f"{prob2:.2f}%",
                        'Difference': f"{abs(prob1 - prob2):.2f}%"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)
                
                # Show agreement or disagreement
                if predictions1[0]['class'] == predictions2[0]['class']:
                    st.success(f"Both models agree on the prediction: **{predictions1[0]['class']}**")
                else:
                    st.warning(f"Models disagree: Model 1 predicts **{predictions1[0]['class']}** while Model 2 predicts **{predictions2[0]['class']}**")

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
            # Get predictions from both models
            test_results1 = predict_image(image, "model1", is_path=True)
            test_results2 = predict_image(image, "model2", is_path=True)
            
            # Display test results for model 1
            st.sidebar.subheader(f"{MODEL_CONFIGS['model1']['name']} Results:")
            if 'error' in test_results1[0]:
                st.sidebar.error(test_results1[0]['error'])
            else:
                for result in test_results1:
                    st.sidebar.write(f"- {result['class']}: {result['probability']:.2f}%")
                if test_results1 and 'inference_time' in test_results1[0]:
                    st.sidebar.info(f"Inference time: {test_results1[0]['inference_time']:.4f} seconds")
            
            # Display test results for model 2
            st.sidebar.subheader(f"{MODEL_CONFIGS['model2']['name']} Results:")
            if 'error' in test_results2[0]:
                st.sidebar.error(test_results2[0]['error'])
            else:
                for result in test_results2:
                    st.sidebar.write(f"- {result['class']}: {result['probability']:.2f}%")
                if test_results2 and 'inference_time' in test_results2[0]:
                    st.sidebar.info(f"Inference time: {test_results2[0]['inference_time']:.4f} seconds")
    except Exception as e:
        st.sidebar.error(f"Error processing image: {e}")

# Add information about the models
st.sidebar.header("About")
st.sidebar.info("""
This application compares two different CNN models for classifying weather conditions in images.
The models can identify weather types: cloudy, foggy, rainy, shine, and sunrise.
""")

# Model details
st.sidebar.header("Model Details")
st.sidebar.subheader(f"Model 1: {MODEL_CONFIGS['model1']['name']}")
st.sidebar.write(f"Architecture: {MODEL_CONFIGS['model1']['architecture']}")
st.sidebar.write(f"Source: Hugging Face ({MODEL_CONFIGS['model1']['repo_id']})")

st.sidebar.subheader(f"Model 2: {MODEL_CONFIGS['model2']['name']}")
st.sidebar.write(f"Architecture: {MODEL_CONFIGS['model2']['architecture']}")
st.sidebar.write(f"Source: Hugging Face ({MODEL_CONFIGS['model2']['repo_id']})")

st.sidebar.write(f"Classes: {', '.join(LABELS)}")