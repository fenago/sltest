import streamlit as st
import cv2
import numpy as np

# Write the title of the app
st.title("Image Thresholding App")

# Load the image
img_path = './road_lanes.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    # Perform binary thresholding
    retval, img_thresh = cv2.threshold(img, 165, 255, cv2.THRESH_BINARY)

    # Display the original and thresholded images side-by-side
    st.image([img, img_thresh], caption=['Original Image', 'Thresholded Image'], width=300)
else:
    st.error("Error: Could not load the image.")
