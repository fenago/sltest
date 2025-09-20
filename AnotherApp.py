import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="CV Lab - Week 4", layout="wide")

# Sidebar with logo
with st.sidebar:
    try:
        response = requests.get("https://www.mdc.edu/about/img/west-campus.jpg")
        img = Image.open(BytesIO(response.content))
        st.image(img, use_column_width=True)
    except:
        pass
    
    st.title("CV Week 4")
    mode = st.radio("Select Mode", ["Thresholding", "Logical Operations"])

st.title("Computer Vision Lab - Week 4")

if mode == "Thresholding":
    st.header("Image Thresholding")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded:
            image = Image.open(uploaded)
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            st.image(gray, caption="Original", use_column_width=True, clamp=True)
            
            # Global threshold controls
            st.subheader("Global Thresholding")
            thresh_val = st.slider("Threshold Value", 0, 255, 127)
            thresh_type = st.selectbox("Type", 
                ["THRESH_BINARY", "THRESH_BINARY_INV"])
            
            # Apply global threshold
            if thresh_type == "THRESH_BINARY":
                _, global_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            else:
                _, global_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            with col2:
                st.image(global_thresh, caption=f"Global Threshold ({thresh_val})", 
                        use_column_width=True, clamp=True)
                
                # Adaptive threshold controls
                st.subheader("Adaptive Thresholding")
                block_size = st.slider("Block Size", 3, 99, 11, step=2)
                c_value = st.slider("C Value", -10, 10, 2)
                
                # Apply adaptive threshold
                adaptive = cv2.adaptiveThreshold(gray, 255, 
                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 
                    block_size, c_value)
                
                st.image(adaptive, caption=f"Adaptive (Block={block_size}, C={c_value})", 
                        use_column_width=True, clamp=True)

else:  # Logical Operations
    st.header("Logical Operations")
    
    col1, col2, col3 = st.columns(3)
    
    # Create simple shapes
    img_rect = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img_rect, (50, 50), (150, 150), 255, -1)
    
    img_circle = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img_circle, (100, 100), 60, 255, -1)
    
    with col1:
        st.image(img_rect, caption="Rectangle", use_column_width=True)
    
    with col2:
        st.image(img_circle, caption="Circle", use_column_width=True)
    
    with col3:
        operation = st.selectbox("Operation", ["AND", "OR", "XOR", "NOT"])
        
        if st.button("Apply"):
            if operation == "AND":
                result = cv2.bitwise_and(img_rect, img_circle)
            elif operation == "OR":
                result = cv2.bitwise_or(img_rect, img_circle)
            elif operation == "XOR":
                result = cv2.bitwise_xor(img_rect, img_circle)
            else:  # NOT
                result = cv2.bitwise_not(img_rect)
            
            st.image(result, caption=f"Result: {operation}", use_column_width=True)
