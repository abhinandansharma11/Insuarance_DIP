# 📄 Insurance Document Intelligence & OCR Pipeline (DIP Project 004)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-yellow)

An end-to-end Digital Image Processing (DIP) and Computer Vision pipeline designed to extract critical key-value pairs (Policy Numbers, Claim Amounts, Dates) from highly degraded insurance documents, invoices, and ID cards.

## 🚀 Features
* **Adaptive Dual-Path Architecture:** Intelligently bypasses aggressive filters for pristine digital PDFs while applying heavy restoration to degraded mobile camera scans.
* **Geometric Deskewing:** Utilizes Canny edge detection and Homography to flatten tilted, off-angle documents.
* **Illumination Normalization:** Applies frequency-domain homomorphic filtering and Gamma correction to neutralize severe shadows and glare.
* **Polarity-Safe Morphology:** A custom automated inversion check ensures black-on-white text is never erased during structural dilation/closing.
* **Tesseract OCR Integration:** Highly optimized regex hierarchy for parsing real-world insurance formats (e.g., `SLI/HLT/2026/...`).
* **Interactive UI:** A beautifully styled Streamlit dashboard for real-time testing, parameter tuning, and pipeline visualization.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Vision & Processing:** OpenCV, NumPy
* **OCR Engine:** Tesseract 5.0
* **Frontend Dashboard:** Streamlit
* **Evaluation:** Pandas, Scikit-Image (CER, WER, SSIM)

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/abhinandansharma11/Insuarance_DIP.git](https://github.com/abhinandansharma11/Insuarance_DIP.git)
cd Insuarance_DIP
