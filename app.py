from venv import logger
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
from rembg import remove
from io import BytesIO
import time
import torch
import logging

# ========== PAGE CONFIG ========== #
st.set_page_config(
    page_title="PhotoPerfect AI - Professional Photo Enhancer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "### 🚀 Powered by AI\nProfessional-grade photo enhancement"
    }
)

# ========== CUSTOM CSS ========== #
def load_css():
    st.markdown(f"""
    <style>
        /* Main container */
        .stApp {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(195deg, #2c3e50 0%, #1a2b3c 100%) !important;
            border-right: 1px solid #4a5568;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50 !important;
        }}
        
        /* Buttons */
        .stButton>button {{
            background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            transition: all 0.3s !important;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
        }}
        
        /* File uploader */
        [data-testid="stFileUploader"] {{
            border: 2px dashed #4facfe !important;
            border-radius: 12px !important;
            padding: 30px !important;
            background: rgba(255, 255, 255, 0.7) !important;
        }}
        
        /* Images */
        .stImage>img {{
            border-radius: 12px !important;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1) !important;
            transition: transform 0.3s, box-shadow 0.3s !important;
        }}
        
        .stImage>img:hover {{
            transform: scale(1.01);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15) !important;
        }}
        
        /* Slider */
        .stSlider [data-testid="stTickBar"] {{
            background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%) !important;
        }}
        
        .stSlider [data-testid="stTickBar"]>div {{
            background: white !important;
        }}
        
        /* Progress bar */
        .stProgress>div>div>div {{
            background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%) !important;
        }}
        
        /* Cards */
        .card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }}
    </style>
    """, unsafe_allow_html=True)

load_css()

# ========== MODEL PATHS ========== #
MODEL_PATHS = {
    'GFPGAN': 'GFPGANv1.4.pth',
    'RealESRGAN': 'RealESRGAN_x4plus.pth'
}

# ========== IMAGE ENHANCER CLASS ========== #
class ImageEnhancer:
    def __init__(self):
        self.gfpgan_model = None
        self.upsampler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models_loaded = False
        
    def load_models(self):
        """Load pre-downloaded models"""
        if self.models_loaded:
            return
            
        try:
            # Verify model files exist
            for model_name, path in MODEL_PATHS.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{model_name} model not found at {path}")
            
            # Load GFPGAN for face enhancement
            from gfpgan import GFPGANer
            self.gfpgan_model = GFPGANer(
                model_path=MODEL_PATHS['GFPGAN'],
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                device=self.device
            )
            
            # Load RealESRGAN for super resolution
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=MODEL_PATHS['RealESRGAN'],
                model=model,
                device=self.device
            )
            
            self.models_loaded = True
            logger.info("AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            st.error(f"Model loading failed: {str(e)}")

    def enhance_image(self, image: Image.Image, enhancement_type: str, strength: int = 5) -> np.ndarray:
        """Apply enhancements with progress tracking"""
        self.load_models()
        img_array = np.array(image)
        
        with st.spinner(f"✨ Applying {enhancement_type} enhancement..."):
            try:
                if enhancement_type == "Super Resolution":
                    if self.upsampler:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        img_array = img_array.astype(np.float32) / 255.0
                        output, _ = self.upsampler.enhance(img_array, outscale=4)
                        output = (output * 255).clip(0, 255).astype(np.uint8)
                        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                    else:
                        height, width = img_array.shape[:2]
                        output = cv2.resize(img_array, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
                
                elif enhancement_type == "Face Enhancement":
                    if self.gfpgan_model:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        _, _, output = self.gfpgan_model.enhance(
                            img_array,
                            has_aligned=False,
                            only_center_face=False,
                            paste_back=True,
                            weight=min(0.5 + strength/20, 0.9)  # Dynamic weight based on slider
                        )
                        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                    else:
                        output = self._advanced_face_enhancement(img_array)
                
                elif enhancement_type == "Denoise":
                    output = cv2.fastNlMeansDenoisingColored(img_array, None, strength, strength, 7, 21)
                
                elif enhancement_type == "Sharpen":
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    output = cv2.filter2D(img_array, -1, kernel)
                
                elif enhancement_type == "Remove Background":
                    output = remove(img_array)
                
                elif enhancement_type == "Color Correction":
                    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    output = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                
                else:
                    output = img_array
                
                return output
                
            except Exception as e:
                logger.error(f"Enhancement failed: {str(e)}")
                st.error(f"Enhancement failed: {str(e)}")
                return img_array

    def _advanced_face_enhancement(self, img_array: np.ndarray) -> np.ndarray:
        """Advanced face enhancement fallback"""
        # Contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        output = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        output = cv2.filter2D(output, -1, kernel)
        
        # Noise reduction
        output = cv2.bilateralFilter(output, 9, 75, 75)
        
        return output

# ========== MAIN APP ========== #
def main():
    # Initialize enhancer
    enhancer = ImageEnhancer()
    
    # App header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #2c3e50; font-weight: 700;">✨ PhotoPerfect AI</h1>
        <p style="font-size: 1.1rem; color: #4a5568;">
            Professional-grade photo enhancement powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: white;">⚙️ Settings</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model status card
        with st.container():
            st.markdown("""
            <div class="card">
                <h4 style="color: #2c3e50; margin-top: 0;">Model Status</h4>
            """, unsafe_allow_html=True)
            
            if all(os.path.exists(path) for path in MODEL_PATHS.values()):
                st.success("✅ All AI models loaded")
                if torch.cuda.is_available():
                    st.info(f"⚡ Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    st.warning("🐢 Using CPU (slower)")
            else:
                st.error("❌ Some models missing")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Enhancement options
        with st.container():
            st.markdown("""
            <div class="card">
                <h4 style="color: #2c3e50; margin-top: 0;">Enhancement Options</h4>
            """, unsafe_allow_html=True)
            
            enhancement_type = st.selectbox(
                "Enhancement Type",
                [
                    "Face Enhancement", 
                    "Super Resolution", 
                    "Denoise", 
                    "Sharpen", 
                    "Remove Background", 
                    "Color Correction"
                ],
                index=0
            )
            
            strength = st.slider("Enhancement Strength", 1, 10, 7)
            auto_download = st.checkbox("Auto-download result", True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # Upload card
        with st.container():
            st.markdown("""
            <div class="card">
                <h3 style="color: #2c3e50; margin-top: 0;">📤 Original Image</h3>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"],
                key="file_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if uploaded_file:
            # Enhanced image card
            with st.container():
                st.markdown("""
                <div class="card">
                    <h3 style="color: #2c3e50; margin-top: 0;">✨ Enhanced Image</h3>
                """, unsafe_allow_html=True)
                
                enhanced = enhancer.enhance_image(original_image, enhancement_type, strength)
                
                if enhanced is not None:
                    st.image(enhanced, use_container_width=True)
                    
                    # Download button
                    buf = BytesIO()
                    enhanced_pil = Image.fromarray(enhanced)
                    enhanced_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.download_button(
                            label="📥 Download Enhanced Image",
                            data=byte_im,
                            file_name=f"enhanced_{enhancement_type.lower().replace(' ', '_')}.png",
                            mime="image/png",
                            type="primary",
                            use_container_width=True
                        )
                    
                    with col_b:
                        if st.button("🔄 Enhance Again", use_container_width=True):
                            st.rerun()
                    
                    # Comparison section
                    st.markdown("---")
                    st.markdown("### 🔍 Before/After Comparison")
                    st.image([original_image, enhanced], caption=["Original", "Enhanced"], width=300)
                    
                    # Enhancement metrics
                    if enhancement_type == "Super Resolution":
                        orig_size = original_image.size
                        enh_size = enhanced_pil.size
                        st.metric(
                            "🖼️ Resolution Boost", 
                            f"{orig_size[0]}×{orig_size[1]} → {enh_size[0]}×{enh_size[1]}",
                            delta=f"+{(enh_size[0]*enh_size[1])/(orig_size[0]*orig_size[1]):.1f}x"
                        )
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Placeholder card
            with st.container():
                st.markdown("""
                <div class="card" style="text-align: center; padding: 40px 20px;">
                    <h3 style="color: #4a5568;">👆 Upload an image to see magic!</h3>
                    <p style="color: #718096;">Your enhanced photo will appear here</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()