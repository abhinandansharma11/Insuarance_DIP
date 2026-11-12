import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def generate_report_figures():
    """Generates and saves side-by-side intermediate pipeline stages for report generation."""
    os.makedirs("data/figures", exist_ok=True)
    
    # Grab a sample from synthetic dataset if available
    img_dir = "data/synthetic_dataset/images"
    if not os.path.exists(img_dir) or not os.listdir(img_dir):
        print("Please run data generation or experiments first to populate images.")
        return

    sample_filename = os.listdir(img_dir)[0]
    sample_path = os.path.join(img_dir, sample_filename)
    orig = cv2.imread(sample_path)

    # Simulate pipeline steps for figure rendering
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("(a) Degraded Input Document")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(clahe, cmap='gray')
    axes[0, 1].set_title("(b) CLAHE Contrast Enhancement")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(otsu, cmap='gray')
    axes[1, 0].set_title("(c) Otsu Binarization")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(closed, cmap='gray')
    axes[1, 1].set_title("(d) Morphological Cleaning")
    axes[1, 1].axis('off')

    plt.tight_layout()
    fig_path = "data/figures/pipeline_stages_comparison.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Report figure successfully generated and saved to {fig_path}")

if __name__ == "__main__":
    generate_report_figures()