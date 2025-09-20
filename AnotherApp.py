import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Computer Vision Playground",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        font-size: 3rem;
        color: white;
        text-align: center;
        padding: 2rem;
        background: rgba(0,0,0,0.3);
        border-radius: 10px;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .sub-header {
        font-size: 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .stButton>button {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# Sidebar
with st.sidebar:
    # Logo
    st.markdown("### 🎓 Miami Dade College")
    try:
        response = requests.get("https://www.mdc.edu/about/img/west-campus.jpg")
        img = Image.open(BytesIO(response.content))
        st.image(img, use_column_width=True)
    except:
        st.info("MDC Logo")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🎯 Navigation")
    app_mode = st.selectbox(
        "Choose Application",
        ["🏠 Home", "🎨 Image Thresholding", "🔧 Logical Operations", "📚 Tutorial"]
    )
    
    st.markdown("---")
    
    # Fun facts
    st.markdown("### 💡 Fun Fact")
    facts = [
        "Computer vision helps self-driving cars see the road!",
        "Thresholding is used in medical imaging to detect tumors.",
        "Binary images use only 1 bit per pixel!",
        "OpenCV processes over 1 billion images daily worldwide.",
        "The first digital image was created in 1957!"
    ]
    import random
    st.info(facts[random.randint(0, len(facts)-1)])
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Created with ❤️")
    st.markdown("Computer Vision Week 4")

# Main content
st.markdown('<h1 class="main-header">🎨 Computer Vision Playground</h1>', unsafe_allow_html=True)

if app_mode == "🏠 Home":
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("## Welcome to the Interactive CV Lab! 🚀")
        st.markdown("""
        This application lets you explore fundamental computer vision concepts through interactive demos:
        
        ### What You Can Do:
        - **🎨 Image Thresholding**: Convert grayscale images to binary
        - **🔧 Logical Operations**: Combine images using AND, OR, XOR, NOT
        - **📚 Learn**: Understand the theory behind each operation
        
        ### Key Skills You'll Master:
        - Apply global and adaptive thresholding
        - Create binary masks for image processing
        - Combine images using bitwise operations
        - Build foundations for lane detection and OCR
        
        **Select an option from the sidebar to begin!**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Animation effect
        st.balloons()

elif app_mode == "🎨 Image Thresholding":
    st.markdown('<h2 class="sub-header">Image Thresholding Laboratory</h2>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🎯 Global Thresholding", "🔬 Adaptive Thresholding"])
    
    with tab1:
        col1, col2 = st.columns([1,1])
        with col1:
            uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
            
            # Sample images
            st.markdown("#### Or try sample images:")
            sample = st.radio(
                "Select a sample:",
                ["None", "Road Lanes", "Sheet Music", "Text Document"]
            )
            
            if sample == "Road Lanes":
                # Create a sample road lanes image
                img = np.zeros((300, 400), dtype=np.uint8)
                cv2.line(img, (50, 250), (150, 50), 200, 5)
                cv2.line(img, (250, 250), (350, 50), 200, 5)
                cv2.rectangle(img, (0, 200), (400, 300), 100, -1)
                st.session_state.original_image = img
                
            elif sample == "Sheet Music":
                # Create a sample sheet music with shadows
                img = np.ones((300, 400), dtype=np.uint8) * 255
                # Add shadows
                img[50:150, 0:100] = img[50:150, 0:100] * 0.7
                # Add music notes (simplified)
                for i in range(5):
                    cv2.line(img, (50, 100 + i*20), (350, 100 + i*20), 0, 2)
                cv2.circle(img, (100, 110), 5, 0, -1)
                cv2.circle(img, (150, 120), 5, 0, -1)
                cv2.circle(img, (200, 100), 5, 0, -1)
                st.session_state.original_image = img
                
            elif sample == "Text Document":
                # Create sample text
                img = np.ones((300, 400), dtype=np.uint8) * 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'Computer Vision', (50, 100), font, 1, 0, 2)
                cv2.putText(img, 'is Amazing!', (100, 200), font, 1, 0, 2)
                # Add some noise
                noise = np.random.normal(0, 10, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
                st.session_state.original_image = img
        
        with col2:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img_array
                st.session_state.original_image = img_gray
                st.image(img_gray, caption="Original Image", use_column_width=True, clamp=True)
            elif hasattr(st.session_state, 'original_image'):
                st.image(st.session_state.original_image, caption="Sample Image", use_column_width=True, clamp=True)
    
    with tab2:
        if hasattr(st.session_state, 'original_image'):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🎚️ Threshold Settings")
                
                threshold_value = st.slider(
                    "Threshold Value",
                    min_value=0,
                    max_value=255,
                    value=127,
                    step=1,
                    help="Pixels above this value become white"
                )
                
                max_value = st.slider(
                    "Max Value",
                    min_value=0,
                    max_value=255,
                    value=255,
                    step=1
                )
                
                threshold_type = st.selectbox(
                    "Threshold Type",
                    ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"]
                )
                
                # Apply thresholding
                thresh_types = {
                    "THRESH_BINARY": cv2.THRESH_BINARY,
                    "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
                    "THRESH_TRUNC": cv2.THRESH_TRUNC,
                    "THRESH_TOZERO": cv2.THRESH_TOZERO,
                    "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV
                }
                
                if st.button("🎨 Apply Global Thresholding", key="global"):
                    _, thresh_img = cv2.threshold(
                        st.session_state.original_image,
                        threshold_value,
                        max_value,
                        thresh_types[threshold_type]
                    )
                    st.session_state.processed_image = thresh_img
            
            with col2:
                if st.session_state.processed_image is not None:
                    # Display comparison
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.imshow(st.session_state.original_image, cmap='gray')
                    ax1.set_title("Original", fontsize=14, fontweight='bold')
                    ax1.axis('off')
                    
                    ax2.imshow(st.session_state.processed_image, cmap='gray')
                    ax2.set_title(f"Thresholded (T={threshold_value})", fontsize=14, fontweight='bold')
                    ax2.axis('off')
                    
                    st.pyplot(fig)
                    
                    # Histogram
                    st.markdown("#### 📊 Histogram Analysis")
                    fig2, ax = plt.subplots(figsize=(10, 4))
                    hist = cv2.calcHist([st.session_state.original_image], [0], None, [256], [0, 256])
                    ax.plot(hist, color='blue', alpha=0.7, linewidth=2)
                    ax.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold_value}')
                    ax.fill_between(range(256), 0, hist.flatten(), alpha=0.3)
                    ax.set_xlabel("Pixel Value")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Image Histogram with Threshold Line")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig2)
        else:
            st.warning("⚠️ Please upload an image or select a sample first!")
    
    with tab3:
        if hasattr(st.session_state, 'original_image'):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🔬 Adaptive Settings")
                
                adaptive_method = st.selectbox(
                    "Adaptive Method",
                    ["ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C"]
                )
                
                block_size = st.slider(
                    "Block Size",
                    min_value=3,
                    max_value=99,
                    value=11,
                    step=2,
                    help="Size of pixel neighborhood (must be odd)"
                )
                
                c_value = st.slider(
                    "C (Constant)",
                    min_value=-20,
                    max_value=20,
                    value=2,
                    step=1,
                    help="Constant subtracted from mean"
                )
                
                # Apply adaptive thresholding
                adaptive_methods = {
                    "ADAPTIVE_THRESH_MEAN_C": cv2.ADAPTIVE_THRESH_MEAN_C,
                    "ADAPTIVE_THRESH_GAUSSIAN_C": cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                }
                
                if st.button("🔬 Apply Adaptive Thresholding", key="adaptive"):
                    adaptive_img = cv2.adaptiveThreshold(
                        st.session_state.original_image,
                        255,
                        adaptive_methods[adaptive_method],
                        cv2.THRESH_BINARY,
                        block_size,
                        c_value
                    )
                    st.session_state.processed_image = adaptive_img
            
            with col2:
                if st.session_state.processed_image is not None:
                    # Display comparison
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.imshow(st.session_state.original_image, cmap='gray')
                    ax1.set_title("Original", fontsize=14, fontweight='bold')
                    ax1.axis('off')
                    
                    ax2.imshow(st.session_state.processed_image, cmap='gray')
                    ax2.set_title(f"Adaptive (Block={block_size}, C={c_value})", fontsize=14, fontweight='bold')
                    ax2.axis('off')
                    
                    st.pyplot(fig)
        else:
            st.warning("⚠️ Please upload an image or select a sample first!")

elif app_mode == "🔧 Logical Operations":
    st.markdown('<h2 class="sub-header">Logical Operations Playground</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        st.markdown("### Image A")
        # Create shape A
        shape_a = st.selectbox("Select Shape A", ["Rectangle", "Circle", "Triangle"], key="shape_a")
        img_a = np.zeros((200, 200), dtype=np.uint8)
        
        if shape_a == "Rectangle":
            cv2.rectangle(img_a, (50, 50), (150, 150), 255, -1)
        elif shape_a == "Circle":
            cv2.circle(img_a, (100, 100), 50, 255, -1)
        else:  # Triangle
            pts = np.array([[100, 50], [50, 150], [150, 150]], np.int32)
            cv2.fillPoly(img_a, [pts], 255)
        
        st.image(img_a, caption="Image A", use_column_width=True)
    
    with col2:
        st.markdown("### Image B")
        # Create shape B
        shape_b = st.selectbox("Select Shape B", ["Rectangle", "Circle", "Triangle"], key="shape_b")
        img_b = np.zeros((200, 200), dtype=np.uint8)
        
        if shape_b == "Rectangle":
            cv2.rectangle(img_b, (75, 75), (175, 175), 255, -1)
        elif shape_b == "Circle":
            cv2.circle(img_b, (125, 100), 50, 255, -1)
        else:  # Triangle
            pts = np.array([[125, 75], [75, 175], [175, 175]], np.int32)
            cv2.fillPoly(img_b, [pts], 255)
        
        st.image(img_b, caption="Image B", use_column_width=True)
    
    with col3:
        st.markdown("### Operation")
        operation = st.selectbox(
            "Select Operation",
            ["AND", "OR", "XOR", "NOT (A)", "NOT (B)"]
        )
        
        if st.button("🔧 Apply Operation", key="logical"):
            if operation == "AND":
                result = cv2.bitwise_and(img_a, img_b)
            elif operation == "OR":
                result = cv2.bitwise_or(img_a, img_b)
            elif operation == "XOR":
                result = cv2.bitwise_xor(img_a, img_b)
            elif operation == "NOT (A)":
                result = cv2.bitwise_not(img_a)
            else:  # NOT (B)
                result = cv2.bitwise_not(img_b)
            
            st.image(result, caption=f"Result: {operation}", use_column_width=True)
    
    # Explanation
    st.markdown("---")
    st.markdown("### 📖 Understanding Logical Operations")
    
    with st.expander("Learn More About Each Operation"):
        st.markdown("""
        **AND Operation**: Returns white (255) only where both images have white pixels
        - Used for: Finding intersections, masking operations
        
        **OR Operation**: Returns white where either image has white pixels
        - Used for: Combining regions, union operations
        
        **XOR Operation**: Returns white where images differ
        - Used for: Finding differences, edge detection
        
        **NOT Operation**: Inverts the image (white becomes black, black becomes white)
        - Used for: Creating inverse masks, complementary operations
        """)

elif app_mode == "📚 Tutorial":
    st.markdown('<h2 class="sub-header">Computer Vision Tutorial</h2>', unsafe_allow_html=True)
    
    # Create tutorial sections
    tutorial_topic = st.selectbox(
        "Select Topic",
        ["Introduction to Thresholding", "Types of Thresholding", "Practical Applications", "Code Examples"]
    )
    
    if tutorial_topic == "Introduction to Thresholding":
        st.markdown("""
        ### What is Image Thresholding? 🎯
        
        Image thresholding is a simple yet effective technique for separating objects from the background.
        It converts a grayscale image into a binary image (black and white only).
        
        #### Why Use Thresholding?
        - **Simplification**: Reduces complex images to essential features
        - **Speed**: Binary images are faster to process
        - **Segmentation**: Isolates regions of interest
        - **Preprocessing**: Prepares images for OCR, edge detection, etc.
        
        #### The Basic Concept:
        ```
        if pixel_value > threshold:
            pixel = white (255)
        else:
            pixel = black (0)
        ```
        """)
        
        # Interactive demo
        st.markdown("### Try it yourself!")
        demo_threshold = st.slider("Adjust threshold:", 0, 255, 127)
        
        # Create gradient image
        gradient = np.tile(np.arange(0, 256, 1), (100, 1)).astype(np.uint8)
        _, binary = cv2.threshold(gradient, demo_threshold, 255, cv2.THRESH_BINARY)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
        ax1.imshow(gradient, cmap='gray', aspect='auto')
        ax1.set_title("Original Gradient")
        ax1.axis('off')
        
        ax2.imshow(binary, cmap='gray', aspect='auto')
        ax2.set_title(f"After Thresholding (T={demo_threshold})")
        ax2.axis('off')
        
        st.pyplot(fig)
    
    elif tutorial_topic == "Types of Thresholding":
        st.markdown("""
        ### Global vs Adaptive Thresholding 🔍
        
        #### Global Thresholding
        - Uses a **single threshold value** for the entire image
        - Best for images with uniform lighting
        - Fast and simple
        - Example: `cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)`
        
        #### Adaptive Thresholding
        - Calculates threshold for **each pixel** based on its neighborhood
        - Handles varying lighting conditions
        - More computationally intensive
        - Example: `cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, ...)`
        
        #### Threshold Types:
        1. **THRESH_BINARY**: Standard thresholding
        2. **THRESH_BINARY_INV**: Inverted thresholding
        3. **THRESH_TRUNC**: Truncates at threshold
        4. **THRESH_TOZERO**: Below threshold becomes zero
        5. **THRESH_TOZERO_INV**: Above threshold becomes zero
        """)
    
    elif tutorial_topic == "Practical Applications":
        st.markdown("""
        ### Real-World Applications 🌍
        
        #### 1. Lane Detection 🚗
        - Isolate white/yellow lane markings from road
        - Critical for autonomous vehicles
        - Uses thresholding + edge detection
        
        #### 2. Document Scanning 📄
        - Convert photos to clean black & white documents
        - Removes shadows and lighting variations
        - Adaptive thresholding essential
        
        #### 3. Medical Imaging 🏥
        - Segment tumors, organs, or abnormalities
        - Separate tissue types
        - Often combined with other techniques
        
        #### 4. QR Code Detection 📱
        - Isolate QR code patterns
        - Binary images easier to decode
        - Robust against lighting changes
        
            5. Object Counting 🔢
        - Count cells in microscopy
        - Inventory management
        - Traffic analysis
        """)
    
    elif tutorial_topic == "Code Examples":
        st.markdown("### Python Code Examples 💻")
        
        st.markdown("#### Basic Global Thresholding")
        st.code("""
import cv2
import numpy as np

# Read image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply thresholding
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Display result
cv2.imshow('Thresholded', thresh)
cv2.waitKey(0)
        """, language='python')
        
        st.markdown("#### Adaptive Thresholding")
        st.code("""
# Adaptive thresholding for varying lighting
adaptive_thresh = cv2.adaptiveThreshold(
    img,                             # Source image
    255,                             # Max value
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Method
    cv2.THRESH_BINARY,               # Type
    11,                              # Block size
    2                                # Constant C
)
        """, language='python')
        
        st.markdown("#### Logical Operations")
        st.code("""
# Create masks and combine images
mask = cv2.threshold(logo_gray, 127, 255, cv2.THRESH_BINARY)[1]
mask_inv = cv2.bitwise_not(mask)

# Apply mask to background
background = cv2.bitwise_and(img_bg, img_bg, mask=mask)

# Apply inverse mask to foreground
foreground = cv2.bitwise_and(img_fg, img_fg, mask=mask_inv)

# Combine
result = cv2.add(background, foreground)
        """, language='python')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem;'>
    <p>Made with ❤️ for Computer Vision Learning</p>
    <p>© 2024 MDC Computer Vision Lab</p>
</div>
""", unsafe_allow_html=True)
