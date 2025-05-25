import streamlit as st
import torch
import torch.nn as nn
import tensorflow as tf
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import io
import pandas as pd
from models.cnn_pytorch import get_pretrained_model as get_pytorch_model
from models.cnn_tensorflow import get_pretrained_model as get_tensorflow_model
from models.train_pytorch import Trainer as PyTorchTrainer
from models.train_tensorflow import Trainer as TensorFlowTrainer
from utils import prep_pytorch, prep_tensorflow

# Page configuration
st.set_page_config(
    page_title="🚀 Brain Tumor Classifier - Alex Mutua",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        position: relative;
        z-index: 2;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        position: relative;
        z-index: 2;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .model-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }
    
    .backend-selector {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .upload-zone {
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-pytorch {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
    }
    
    .status-tensorflow {
        background: linear-gradient(45deg, #feca57, #ff9ff3);
        color: white;
    }
    
    .sidebar .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Model configuration
@st.cache_resource
def load_models():
    # Initialize PyTorch model
    pytorch_model = get_pytorch_model(num_classes=4)  # 4 classes for brain tumors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device)
    try:
        pytorch_model.load_state_dict(torch.load("alex.torch", map_location=device))
        pytorch_model.eval()
    except FileNotFoundError:
        st.warning("PyTorch model file 'alex.torch' not found. Using untrained model.")

    # Initialize TensorFlow model
    try:
        tensorflow_model = tf.keras.models.load_model("alex_model.tensorflow")
    except Exception:
        tensorflow_model = get_tensorflow_model(num_classes=4)  # 4 classes for brain tumors
        st.warning("TensorFlow model file 'alex_model.tensorflow' not found. Using untrained model.")

    models = {
        "pytorch": {
            "ResNet34": {
                "model": pytorch_model,
                "description": "Pretrained ResNet34 for brain tumor classification (4 classes)",
                "classes": ["glioma", "meningioma", "notumor", "pituitary"],
                "accuracy": "Unknown (train to evaluate)",
                "params": "~21.3M"
            }
        },
        "tensorflow": {
            "ResNet34/50": {
                "model": tensorflow_model,
                "description": "Pretrained ResNet34 or ResNet50 for brain tumor classification (4 classes)",
                "classes": ["glioma", "meningioma", "notumor", "pituitary"],
                "accuracy": "Unknown (train to evaluate)",
                "params": "~21.3M (ResNet34) or ~25.6M (ResNet50)"
            }
        }
    }
    return models

def preprocess_image(image, backend="pytorch"):
    """Preprocessing tailored to ResNet models (224x224 RGB, ImageNet normalization)"""
    image = image.convert('RGB')  # Ensure RGB
    image = image.resize((224, 224))
    
    if backend == "pytorch":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(image).unsqueeze(0)
        return tensor
    else:  # tensorflow
        img_array = np.array(image).astype('float32') / 255.0
        img_array = tf.keras.applications.resnet.preprocess_input(img_array)
        img_array = img_array.reshape(1, 224, 224, 3)
        return img_array

def predict_pytorch(model, input_tensor):
    """Prediction with PyTorch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        all_probs = probabilities[0].numpy()
    return predicted_class, confidence, all_probs

def predict_tensorflow(model, input_array):
    """Prediction with TensorFlow"""
    predictions = model.predict(input_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    all_probs = predictions[0]
    return predicted_class, confidence, all_probs

def train_model(backend, model, train_data, val_data, lr=0.001, wd=0.0001, epochs=10):
    """Train the model using the provided Trainer classes"""
    if backend == "pytorch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        trainer = PyTorchTrainer(model, train_data, val_data, lr, wd, epochs, device)
        trainer.train(save=True, plot=True)
        accuracy, loss = trainer.evaluate()
        st.success(f"Training completed! Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
    else:  # tensorflow
        trainer = TensorFlowTrainer(model, train_data, val_data, lr, epochs)
        trainer.train(save=True, filename="alex_model.tensorflow", plot=True)
        accuracy, loss = trainer.evaluate()
        st.success(f"Training completed! Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")
    return accuracy, loss

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Brain Tumor Classifier</h1>
        <p>Brain Tumor Classification with PyTorch & TensorFlow - By Alex Mutua</p>
        <div style="margin-top: 1rem;">
            <span class="status-badge status-pytorch">PyTorch Ready</span>
            <span class="status-badge status-tensorflow">TensorFlow Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Backend selection
        st.markdown('<div class="backend-selector">🔧 Choose Backend</div>', unsafe_allow_html=True)
        backend = st.radio(
            "AI Backend:",
            ["pytorch", "tensorflow"],
            format_func=lambda x: f"🔥 PyTorch" if x == "pytorch" else "🟢 TensorFlow",
            help="Select the AI framework to use"
        )
        
        # Model selection
        st.markdown("### 🧠 Model")
        available_models = list(models[backend].keys())
        selected_model = st.selectbox(
            "Architecture:",
            available_models,
            help="Select the model architecture"
        )
        
        # Information about the selected model
        model_info = models[backend][selected_model]
        st.markdown(f"""
        <div class="model-card">
            <h4>📊 {selected_model}</h4>
            <p><strong>Description:</strong> {model_info['description']}</p>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div style="font-size: 1.5em;">🎯</div>
                    <div><strong>{model_info['accuracy']}</strong></div>
                    <div style="font-size: 0.8em;">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 1.5em;">⚡</div>
                    <div><strong>{model_info['params']}</strong></div>
                    <div style="font-size: 0.8em;">Parameters</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Training parameters
        st.markdown("### 🏋️ Training Parameters")
        lr = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        epochs = st.slider("Epochs", 5, 50, 10)
        wd = st.slider("Weight Decay (PyTorch only)", 0.0, 0.001, 0.0001, format="%.5f")
        
        # Training button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training {selected_model} with {backend.upper()}..."):
                try:
                    train_data, val_data = (prep_pytorch.get_data() if backend == "pytorch" 
                                            else prep_tensorflow.get_data())
                    accuracy, loss = train_model(backend, model_info["model"], train_data, val_data, lr, wd, epochs)
                    # Update model accuracy in UI
                    models[backend][selected_model]["accuracy"] = f"{accuracy:.2f}%" if backend == "pytorch" else f"{accuracy * 100:.2f}%"
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
        
        # System information
        st.markdown("### 💻 System")
        device_info = "🟢 CUDA Available" if torch.cuda.is_available() else "🔵 CPU Mode"
        st.markdown(f"**Device:** {device_info}")
        st.markdown(f"**Backend:** {backend.upper()}")

    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload & Configuration")
        
        # Upload zone
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a brain tumor MRI image",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Image preview
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded MRI Image", use_container_width=True)
            
            # Image information
            st.markdown(f"""
            **Image Information:**
            - Format: {image.format}
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Mode: {image.mode}
            """)
    
    with col2:
        st.markdown("### 🎯 Classification Results")
        
        if uploaded_file is not None:
            # Prediction button
            if st.button("🔮 Classify Image", type="primary", use_container_width=True):
                with st.spinner(f"🤖 Classifying with {backend.upper()}..."):
                    try:
                        # Preprocessing based on backend
                        input_data = preprocess_image(image, backend)
                        
                        # Prediction based on backend
                        model = model_info["model"]
                        predicted_class, confidence, all_probs = (predict_pytorch(model, input_data) 
                                                                  if backend == "pytorch" 
                                                                  else predict_tensorflow(model, input_data))
                        
                        # Display main result
                        class_name = model_info["classes"][predicted_class]
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>🎯 Result</h2>
                            <h1 style="font-size: 3em; margin: 0.5rem 0;">{class_name}</h1>
                            <h3>Confidence: {confidence:.1%}</h3>
                            <p style="opacity: 0.8; margin-top: 1rem;">
                                Classifier: {backend.upper()} • Model: {selected_model}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display probability distribution
                        st.markdown("### 📊 Probability Distribution")
                        prob_df = pd.DataFrame({
                            "Class": model_info["classes"],
                            "Probability": [f"{p:.2%}" for p in all_probs]
                        })
                        st.table(prob_df)
                        
                    except Exception as e:
                        st.error(f"❌ Error during classification: {str(e)}")
                        st.info("💡 Try with a clear MRI image or check the format")
        else:
            st.info("👆 Upload an MRI image to start classification")
            
            # Usage tips
            st.markdown("""
            ### 💡 Tips for Best Results
            
            **Recommended Format:**
            - Brain tumor MRI images (e.g., T1-weighted, T2-weighted)
            - RGB format (will be converted automatically)
            - Size: Approximately 224x224 pixels (will be resized)
            
            **Optimal Quality:**
            - High contrast and clear visibility of tumor regions
            - Minimal noise or artifacts
            - Well-aligned images
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px; margin-top: 2rem;">
        <h4>🔬 Brain Tumor Classification Project</h4>
        <p>Developed by <strong>Alex Mutua</strong> • Powered by PyTorch & TensorFlow</p>
        <p style="font-size: 0.9em; opacity: 0.7;">
            Pretrained ResNet models • Cloud-ready deployment • Modern interface
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()