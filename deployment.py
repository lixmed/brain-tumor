import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better appearance with improved color scheme
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498DB;
    }
    
    .sub-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2C3E50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .info-text {
        font-size: 1.1rem;
        color: #34495E;
        line-height: 1.6;
    }
    
    /* Card styling */
    .highlight {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498DB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Prediction result styling */
    .prediction-container {
        text-align: center;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: #FFFFFF;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .glioma-result {
        background-color: #E74C3C;
    }
    
    .meningioma-result {
        background-color: #F39C12;
    }
    
    .notumor-result {
        background-color: #2ECC71;
    }
    
    .pituitary-result {
        background-color: #9B59B6;
    }
    
    /* Chart container */
    .chart-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Info boxes styling */
    .stInfo {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 5px solid #2196F3;
    }
    
    .stSuccess {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 5px solid #4CAF50;
    }
    
    /* Buttons styling */
    .stButton > button {
        background-color: #3498DB;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2980B9;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: #2C3E50;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498DB !important;
        color: white !important;
    }
    
    /* File uploader */
    .uploadedFileData {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 10px;
        background-color: #F8F9FA;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px 0;
        color: #7F8C8D;
        border-top: 1px solid #EAEAEA;
        margin-top: 2rem;
    }
    
    /* Sample images grid */
    .sample-image-container {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        background-color: #F8F9FA;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2C3E50;
    }
    
    /* Make sure text is visible in all containers */
    div[data-testid="stVerticalBlock"] {
        color: #2C3E50;
    }
</style>
""", unsafe_allow_html=True)

# Define class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load the pre-trained model
@st.cache_resource
def load_model_file():
    try:
        model = tf.keras.models.load_model('Densenet.keras')
        return model, None
    except Exception as e:
        return None, str(e)

# Preprocess the uploaded image
def preprocess_image(image):
    # Resize to 244x244 as per the model input
    image = image.resize((244, 244))
    # Convert to array and rescale to [0,1]
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Plot confidence scores as a bar chart
def plot_confidence_scores(predictions):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each class to match our UI theme
    colors = ['#E74C3C', '#F39C12', '#2ECC71', '#9B59B6']
    
    # Create bar plot
    sns.barplot(x=class_labels, y=predictions[0] * 100, palette=colors, ax=ax)
    
    # Customize the plot
    plt.title('Confidence Scores for Each Class', fontsize=16, fontweight='bold', color='#2C3E50')
    plt.xlabel('Class', fontsize=14, color='#2C3E50')
    plt.ylabel('Confidence (%)', fontsize=14, color='#2C3E50')
    plt.xticks(fontsize=12, color='#2C3E50')
    plt.yticks(fontsize=12, color='#2C3E50')
    plt.ylim(0, 100)
    
    # Add value labels on top of bars
    for i, v in enumerate(predictions[0] * 100):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12, color='#2C3E50', fontweight='bold')
    
    # Set grid and style
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust figure appearance
    fig.tight_layout()
    
    return fig

# Main function
def main():
    st.markdown("<h1 class='main-header'>Brain Tumor MRI Classification</h1>", unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Prediction", "ℹ️ About", "❓ Help"])
    
    with tab1:
        st.markdown("<p class='info-text'>Upload an MRI image to classify it as Glioma, Meningioma, Pituitary, or No Tumor.</p>", unsafe_allow_html=True)
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # File uploader with improved container
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            if uploaded_file is not None:
                try:
                    # Display the uploaded image in a nicer container
                    image = Image.open(uploaded_file)
                    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Load model
                    model, error = load_model_file()
                    
                    if model is None:
                        st.error(f"Failed to load model: {error}")
                        st.info("Please make sure the model file 'Densenet.keras' is in the same directory as this app.")
                    else:
                        # Add a prediction button with improved styling
                        st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
                        predict_button = st.button("Analyze MRI Image", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        if predict_button:
                            with st.spinner("Processing image..."):
                                # Simulate processing time
                                time.sleep(1)
                                
                                # Preprocess the image
                                processed_image = preprocess_image(image)
                                
                                # Make prediction
                                predictions = model.predict(processed_image)
                                predicted_class = class_labels[np.argmax(predictions[0])]
                                confidence = np.max(predictions[0]) * 100
                                
                                # Display results in the second column
                                with col2:
                                    st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
                                    
                                    # Display prediction with appropriate styling based on class
                                    st.markdown(f"""
                                    <div class='prediction-container {predicted_class}-result'>
                                        <h2 style='font-size: 2rem; margin-bottom: 0.5rem;'>{predicted_class.upper()}</h2>
                                        <p style='font-size: 1.2rem;'>Confidence: {confidence:.2f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display confidence scores as a bar chart
                                    st.markdown("<h3 class='sub-header'>Confidence Scores</h3>", unsafe_allow_html=True)
                                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                                    fig = plot_confidence_scores(predictions)
                                    st.pyplot(fig)
                                    st.markdown("</div>", unsafe_allow_html=True)
                                    
                                    # Display interpretation
                                    st.markdown("<h3 class='sub-header'>Interpretation</h3>", unsafe_allow_html=True)
                                    if predicted_class == "glioma":
                                        st.markdown("<div class='stInfo'>Gliomas are tumors that occur in the brain and spinal cord. They begin in the glial cells that surround and support nerve cells.</div>", unsafe_allow_html=True)
                                    elif predicted_class == "meningioma":
                                        st.markdown("<div class='stInfo'>Meningiomas are tumors that arise from the meninges — the membranes that surround your brain and spinal cord.</div>", unsafe_allow_html=True)
                                    elif predicted_class == "pituitary":
                                        st.markdown("<div class='stInfo'>Pituitary tumors are abnormal growths that develop in the pituitary gland, which is located at the base of the brain.</div>", unsafe_allow_html=True)
                                    else:
                                        st.markdown("<div class='stSuccess'>No tumor detected in the MRI scan.</div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please ensure the uploaded file is a valid MRI image.")
            else:
                with col2:
                    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                    st.info("Please upload an MRI image to get predictions.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display sample images in a nicer grid with improved styling
                    st.markdown("<h3 class='sub-header'>Sample Images</h3>", unsafe_allow_html=True)
                    st.markdown("<p class='info-text'>The model is trained to recognize these types of brain tumors:</p>", unsafe_allow_html=True)
                    
                    # Create a grid for sample images with better styling
                    cols = st.columns(4)
                    for i, label in enumerate(class_labels):
                        with cols[i]:
                            # Color-code each tumor type to match prediction results
                            if label == "glioma":
                                color = "#E74C3C"
                            elif label == "meningioma":
                                color = "#F39C12"
                            elif label == "notumor":
                                color = "#2ECC71"
                            else:  # pituitary
                                color = "#9B59B6"
                                
                            st.markdown(f"""
                            <div class='sample-image-container'>
                                <h4 style='color: {color}; font-weight: bold;'>{label.capitalize()}</h4>
                                <div style='height: 100px; display: flex; align-items: center; justify-content: center; background-color: {color}30; border-radius: 5px;'>
                                    <span style='color: {color}; font-weight: bold;'>Sample</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h2 class='sub-header'>About This Application</h2>", unsafe_allow_html=True)
        
        # About section with improved styling
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-text'>
        <p>This application uses a deep learning model to classify brain MRI scans into four categories:</p>
        <ul>
            <li><strong style="color: #E74C3C;">Glioma:</strong> A type of tumor that occurs in the brain and spinal cord</li>
            <li><strong style="color: #F39C12;">Meningioma:</strong> A tumor that forms on membranes that cover the brain and spinal cord</li>
            <li><strong style="color: #9B59B6;">Pituitary Tumor:</strong> A growth that develops in the pituitary gland</li>
            <li><strong style="color: #2ECC71;">No Tumor:</strong> Normal brain tissue without any tumor</li>
        </ul>
        <p>The model was trained on thousands of MRI images and achieves high accuracy in classification.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>How It Works</h2>", unsafe_allow_html=True)
        
        # How it works section with better styling
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-text'>
        <p>The application works in the following steps:</p>
        <ol>
            <li>You upload a brain MRI scan image</li>
            <li>The image is preprocessed to match the format expected by the model</li>
            <li>The deep learning model analyzes the image and provides predictions</li>
            <li>Results are displayed with confidence scores for each possible class</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Technology stack
        st.markdown("<h2 class='sub-header'>Technology Stack</h2>", unsafe_allow_html=True)
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-text'>
        <p>This application is built using:</p>
        <ul>
            <li><strong>Streamlit:</strong> For the interactive web interface</li>
            <li><strong>TensorFlow:</strong> For running the deep learning model</li>
            <li><strong>DenseNet:</strong> Neural network architecture for image classification</li>
            <li><strong>Matplotlib & Seaborn:</strong> For data visualization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Help & FAQ</h2>", unsafe_allow_html=True)
        
        # FAQ section with improved styling
        with st.expander("What image formats are supported?"):
            st.markdown("<div class='info-text'>The application supports JPG, JPEG, and PNG image formats.</div>", unsafe_allow_html=True)
        
        with st.expander("How accurate is the model?"):
            st.markdown("<div class='info-text'>The model has been trained on a large dataset of brain MRI scans and achieves high accuracy. However, this tool is for educational purposes only and should not be used for medical diagnosis.</div>", unsafe_allow_html=True)
        
        with st.expander("I'm getting an error when uploading an image"):
            st.markdown("<div class='info-text'>Please ensure that you're uploading a valid image file in JPG, JPEG, or PNG format. The image should be a brain MRI scan for best results.</div>", unsafe_allow_html=True)
        
        with st.expander("The model isn't loading"):
            st.markdown("<div class='info-text'>Make sure the model file 'Densenet.keras' is in the same directory as this application. If the problem persists, try restarting the application.</div>", unsafe_allow_html=True)
        
        with st.expander("Can I use this for medical diagnosis?"):
            st.markdown("<div class='info-text'>No. This application is for educational and demonstration purposes only. It should not be used for medical diagnosis or treatment decisions. Always consult with a qualified healthcare professional for medical advice.</div>", unsafe_allow_html=True)
        
        with st.expander("How do I interpret the results?"):
            st.markdown("""
            <div class='info-text'>
            <p>The application provides:</p>
            <ul>
                <li><strong>Prediction:</strong> The most likely tumor type detected in the MRI image</li>
                <li><strong>Confidence:</strong> How certain the model is about its prediction (higher is better)</li>
                <li><strong>Bar Chart:</strong> Visual representation of confidence scores for all possible classes</li>
            </ul>
            <p>Remember that this is for educational purposes only and should not replace professional medical advice.</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer with better styling
    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.markdown("© 2024 Brain Tumor MRI Classification App | Created with Streamlit", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()