import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="DON Prediction App",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
def get_api_url():
    """Get the API base URL based on environment"""
    # For Vercel deployment
    if os.environ.get("VERCEL_ENV") in ["production", "preview"]:
        return "/api/v1"
    # For local development
    return "http://localhost:8000"

API_URL = get_api_url()

# Utility functions
def check_api_status():
    """Check if the API is reachable"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, "API connected successfully"
        return False, f"API error: {response.status_code}"
    except Exception as e:
        return False, f"API unreachable: {str(e)}"

def make_prediction(features):
    """Make a prediction using the API"""
    try:
        response = requests.post(
            f"{API_URL}/predictions/",
            json={"features": features},
            timeout=10
        )
        if response.status_code == 200:
            return True, response.json()
        return False, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Request failed: {str(e)}"

def get_predictions_history(limit=10):
    """Get prediction history from the API"""
    try:
        response = requests.get(
            f"{API_URL}/predictions/?limit={limit}",
            timeout=10
        )
        if response.status_code == 200:
            return True, response.json()
        return False, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Request failed: {str(e)}"

def format_timestamp(timestamp_str):
    """Format timestamps for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #e6f3ff;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("DON Prediction")
    
    # Check API status
    status, message = check_api_status()
    if status:
        st.success("✅ API Connected")
    else:
        st.error(f"❌ {message}")
    
    # Navigation
    st.markdown("## Navigation")
    page = st.radio("Select Page", ["Home", "Make Prediction", "Prediction History"])
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application predicts Deoxynivalenol (DON) 
    concentration levels in grain samples using a machine learning model.
    """)

# Main content
if page == "Home":
    st.markdown('<div class="main-header">DON Prediction Application</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the DON Prediction application. This tool helps predict Deoxynivalenol (DON) 
    concentration levels in grain samples based on various features.
    
    ### What is DON?
    Deoxynivalenol (DON), also known as vomitoxin, is a mycotoxin produced by Fusarium fungi 
    that commonly affects cereal grains like wheat, barley, and corn. It can cause significant 
    health issues for humans and animals when consumed in contaminated foods.
    
    ### How to Use This Application
    
    1. **Make Prediction**: Upload a CSV file with features or enter feature values manually
    2. **View History**: See recent predictions and their results
    
    ### About the Model
    
    The prediction model was trained on a dataset of grain samples with known DON levels.
    The input features are processed through a neural network that outputs a predicted DON concentration.
    """)
    
    # Display example image or chart
    st.image("https://www.foodsafetynews.com/files/2020/11/wheat-field-grain-640x360.jpg", 
             caption="Wheat field - potential source of DON contamination")
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Note**: This predictive model is intended as a decision support tool and should not replace 
    laboratory testing for regulatory compliance or food safety assurance.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Make Prediction":
    st.markdown('<div class="main-header">Make DON Prediction</div>', unsafe_allow_html=True)
    
    # Choose input method
    input_method = st.radio("Select input method:", ["Upload CSV", "Enter Values Manually"])
    
    if input_method == "Upload CSV":
        st.markdown("""
        Upload a CSV file with features for prediction. The file should contain all 447 required features in the correct order.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file with features:", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Display preview
                st.markdown('<div class="sub-header">Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(df.head())
                
                # Check dimensions
                num_features = df.shape[1]
                st.write(f"Number of features: {num_features}")
                
                if num_features != 447 and not st.checkbox("Override feature count check"):
                    st.warning(f"Expected 447 features, but found {num_features}. Please check your data or override the check.")
                else:
                    if st.button("Make Prediction"):
                        # Use first row for prediction
                        features = df.iloc[0].tolist()
                        
                        with st.spinner("Making prediction..."):
                            success, result = make_prediction(features)
                            
                        if success:
                            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                            st.success(f"Prediction ID: {result['id']}")
                            st.markdown(f"### Predicted DON Value: **{result['don_value']:.2f} ppb**")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"Prediction failed: {result}")
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    else:  # Manual input
        st.markdown("""
        Enter key feature values manually. The remaining features will be set to default values.
        
        **Note**: This is a simplified input for demonstration purposes. For accurate predictions, 
        all 447 features would normally be required.
        """)
        
        # Create column layout for inputs
        col1, col2 = st.columns(2)
        
        # Add some example feature inputs
        with col1:
            feature1 = st.number_input("Feature 1", value=0.1, format="%.2f")
            feature2 = st.number_input("Feature 2", value=-0.2, format="%.2f")
            feature3 = st.number_input("Feature 3", value=0.3, format="%.2f")
        
        with col2:
            feature4 = st.number_input("Feature 4", value=0.4, format="%.2f")
            feature5 = st.number_input("Feature 5", value=0.5, format="%.2f")
            feature6 = st.number_input("Feature 6", value=-0.1, format="%.2f")
        
        if st.button("Make Prediction"):
            # Create a feature vector with the entered values and zeros for the rest
            # Ensuring we have exactly 447 features
            features = [feature1, feature2, feature3, feature4, feature5, feature6] + [0.0] * 441
            
            with st.spinner("Making prediction..."):
                success, result = make_prediction(features)
                
            if success:
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.success(f"Prediction ID: {result['id']}")
                st.markdown(f"### Predicted DON Value: **{result['don_value']:.2f} ppb**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Prediction failed: {result}")

elif page == "Prediction History":
    st.markdown('<div class="main-header">Prediction History</div>', unsafe_allow_html=True)
    
    # Number of records to show
    limit = st.slider("Number of records to show", min_value=5, max_value=50, value=10)
    
    if st.button("Refresh Data"):
        st.session_state.refresh_time = time.time()
    
    # Get prediction history
    with st.spinner("Loading prediction history..."):
        success, history = get_predictions_history(limit)
    
    if success:
        if not history:
            st.info("No prediction history found. Make some predictions first!")
        else:
            # Create a dataframe for easier display
            records = []
            for pred in history:
                record = {
                    "ID": pred["id"],
                    "DON Value (ppb)": round(pred["prediction"], 2),
                    "Timestamp": format_timestamp(pred["created_at"]),
                    "Features Sample": str(pred["input_data"][:3]) + "..."
                }
                records.append(record)
            
            history_df = pd.DataFrame(records)
            st.dataframe(history_df, use_container_width=True)
            
            # Display individual records with expandable details
            st.markdown('<div class="sub-header">Detailed Records</div>', unsafe_allow_html=True)
            for i, pred in enumerate(history):
                with st.expander(f"Prediction {i+1}: {format_timestamp(pred['created_at'])} - DON: {pred['prediction']:.2f} ppb"):
                    st.write("Prediction ID:", pred["id"])
                    st.write("DON Value:", f"{pred['prediction']:.2f} ppb")
                    st.write("Time:", format_timestamp(pred["created_at"]))
                    
                    # Display first few features
                    st.write("Input Features (first 10):")
                    features_preview = pd.DataFrame({
                        "Index": range(10),
                        "Value": pred["input_data"][:10]
                    })
                    st.dataframe(features_preview)
    else:
        st.error(f"Failed to load prediction history: {history}")

# Footer
st.markdown("---")
st.markdown("© 2025 DON Prediction Application | Created for ML Pipeline Project")