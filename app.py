import streamlit as st
import cv2
import numpy as np

from src.preprocessing import DocumentPreprocessor
from src.frequency_filters import FrequencyDomainFilter
from src.segmentation import DocumentSegmentation
from src.ocr_extraction import OCRExtractor

st.set_page_config(page_title="Insurance Document Intelligence", layout="wide")

st.markdown("""
    <style>
    .hero-card { background: #ffffff; padding: 2.5rem; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); text-align: center; border: 1px solid #e2e8f0; margin-bottom: 2rem;}
    .metric-card { background: #ffffff; padding: 1.2rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.04); border-left: 5px solid #2563eb; border: 1px solid #e2e8f0;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #2563eb;'>📄 Insurance Document Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Digital Image Processing Pipeline</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2>⚙️ Pipeline Tuning</h2>", unsafe_allow_html=True)
doc_type = st.sidebar.radio("Select Input Condition:", ["Clean Digital Document (PDF/Screenshot)", "Degraded Camera Scan (Shadows/Blur)"])
gamma_val = st.sidebar.slider("Gamma Correction (Only applies to Camera Scans)", 0.5, 2.0, 1.0, 0.1)

st.markdown("<div class='hero-card'><h3>Upload Document</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    orig_img = cv2.imdecode(file_bytes, 1)

    preprocessor = DocumentPreprocessor()
    freq_filter = FrequencyDomainFilter()
    segmenter = DocumentSegmentation()
    ocr_extractor = OCRExtractor()

    with st.spinner("Executing Pipeline..."):
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

        # Smart Bypass logic
        if doc_type == "Clean Digital Document (PDF/Screenshot)":
            binary = segmenter.apply_global_binarization(gray)
            filtered_comps = segmenter.apply_morphology(binary, operation="closing")
        else:
            pts = preprocessor.detect_document_contour(orig_img)
            warped = preprocessor.correct_perspective(orig_img, pts)
            homomorphic = freq_filter.apply_homomorphic_filter(warped)
            enhanced_gamma = preprocessor.adjust_gamma(homomorphic, gamma=gamma_val)
            binary = segmenter.apply_global_binarization(enhanced_gamma)
            filtered_comps = segmenter.apply_morphology(binary, operation="closing")

        baseline_text = ocr_extractor.run_baseline_ocr(orig_img)
        proposed_text = ocr_extractor.run_enhanced_ocr(filtered_comps)
        extracted_fields = ocr_extractor.extract_key_fields(proposed_text)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📥 Original Input")
        st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown("### 🔍 Processed DIP Output")
        st.image(filtered_comps, use_container_width=True, clamp=True)

    st.markdown("---")
    st.markdown("<h2>📊 Extracted Entities</h2>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Document Type", extracted_fields.get("document_type", "UNKNOWN").upper())
    m_col2.metric("Policy Number", extracted_fields.get("policy_number", "NOT_FOUND"))
    m_col3.metric("Claim/Sum Amount", extracted_fields.get("claim_amount", "NOT_FOUND"))

    st.json(extracted_fields)
    with st.expander("📝 View Raw OCR vs. DIP-Enhanced OCR Comparison Text"):
        col_ocr1, col_ocr2 = st.columns(2)
        with col_ocr1:
            st.markdown("**Baseline OCR (Raw Input):**")
            st.text(baseline_text)
        with col_ocr2:
            st.markdown("**Proposed DIP Pipeline OCR:**")
            st.text(proposed_text)