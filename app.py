import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

# Set page design
st.set_page_config(page_title="Tomato Leaf Disease Diagnosis", page_icon="🌿", layout="centered")

st.title("🌿 Tomato Leaf Disease Classification System")
st.write("Upload an image of a tomato plant leaf to diagnose potential diseases using our optimized MobileNetV2 model.")

# Class names matching dataset structure
class_names = [
    'Tomato - Bacterial spot', 'Tomato - Early blight', 'Tomato - Late blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria leaf spot', 
    'Tomato - Spider mites (Two-spotted spider mite)', 'Tomato - Target Spot', 
    'Tomato - Yellow Leaf Curl Virus', 'Tomato - Tomato mosaic virus', 
    'Tomato - Healthy'
]

@st.cache_resource
def load_model():
    model = torchvision.models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    MODEL_PATH = "tomato_model_final.pth"
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

model = load_model()

if model is None:
    st.error("❌ Could not find 'tomato_model_final.pth' weights file in this folder. Please verify the file exists.")
else:
    # Image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Leaf Image', width=300)
        
        st.write("🔄 Analyzing structure and symptoms...")
        
        # Run inference
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, prediction = torch.max(probabilities, 0)
            predicted_class = class_names[prediction.item()]
            
        # Display Results
        st.subheader("📊 Diagnosis Result:")
        if "Healthy" in predicted_class:
            st.success(f"*Result: {predicted_class}* (Confidence: {confidence.item()*100:.2f}%)")
        else:
            st.warning(f"*Detected: {predicted_class}* (Confidence: {confidence.item()*100:.2f}%)")