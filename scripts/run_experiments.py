import os
import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from src.data_generator import InsuranceDataGenerator
from src.preprocessing import DocumentPreprocessor
from src.frequency_filters import FrequencyDomainFilter
from src.segmentation import DocumentSegmentation
from src.ocr_extraction import OCRExtractor
from src.metrics import EvaluationMetrics

def run_full_evaluation_pipeline():
    """Generates synthetic dataset, executes baseline vs proposed DIP pipeline, and saves metrics."""
    print("=== INITIALIZING PIPELINE EXPERIMENTS ===")
    
    # 1. Generate Dataset
    generator = InsuranceDataGenerator()
    preprocessor = DocumentPreprocessor()
    freq_filter = FrequencyDomainFilter()
    segmenter = DocumentSegmentation()
    ocr_extractor = OCRExtractor()
    metrics_calc = EvaluationMetrics()

    num_samples = 15
    image_paths = []
    ground_truths = []

    print(f"Generating {num_samples} synthetic insurance documents...")
    for i in range(num_samples):
        doc_type = np.random.choice(["claim_form", "invoice", "id_card"])
        img_path, gt_path, gt_data = generator.generate_synthetic_document(doc_type, i)
        image_paths.append(img_path)
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truths.append(json.load(f))

    results = []

    print("Running Baseline vs Proposed DIP Pipeline comparison...")
    for idx, (img_path, gt) in enumerate(tqdm(zip(image_paths, ground_truths), total=num_samples)):
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue

        # --- BASELINE PIPELINE ---
        baseline_text = ocr_extractor.run_baseline_ocr(orig_img)
        baseline_fields = ocr_extractor.extract_key_fields(baseline_text)
        
        # Build reference string from GT
        ref_string = f"{gt['policy_number']} {gt['claim_amount']} {gt['name']} {gt['date']}"
        base_err = metrics_calc.calculate_text_error_rates(ref_string, baseline_text)
        base_f1 = metrics_calc.calculate_extraction_f1(gt, baseline_fields)

        # --- PROPOSED DIP PIPELINE ---
        # 1. Perspective correction
        pts = preprocessor.detect_document_contour(orig_img)
        warped = preprocessor.correct_perspective(orig_img, pts)

        # 2. Illumination & Denoising
        homomorphic = freq_filter.apply_homomorphic_filter(warped)
        enhanced_gamma = preprocessor.adjust_gamma(homomorphic, gamma=1.3)

        # 3. Binarization & Morphology
        binary = segmenter.apply_local_binarization(enhanced_gamma, method="wolf", window_size=15)
        cleaned_morph = segmenter.apply_morphology(binary, operation="closing")
        filtered_comps = segmenter.filter_connected_components(cleaned_morph)

        # 4. OCR & Parsing
        proposed_text = ocr_extractor.run_enhanced_ocr(filtered_comps)
        proposed_fields = ocr_extractor.extract_key_fields(proposed_text)

        prop_err = metrics_calc.calculate_text_error_rates(ref_string, proposed_text)
        prop_f1 = metrics_calc.calculate_extraction_f1(gt, proposed_fields)
        img_quality = metrics_calc.calculate_image_quality(orig_img, filtered_comps)

        results.append({
            "Sample": f"Sample_{idx}",
            "Type": gt["document_type"],
            "Baseline_CER": base_err["CER"],
            "Proposed_CER": prop_err["CER"],
            "Baseline_WER": base_err["WER"],
            "Proposed_WER": prop_err["WER"],
            "Baseline_F1": base_f1,
            "Proposed_F1": prop_f1,
            "PSNR": img_quality["PSNR"],
            "SSIM": img_quality["SSIM"]
        })

    # Export CSV
    os.makedirs("data/results", exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = "data/results/experiment_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nExperiments completed successfully! Results saved to {csv_path}")
    print("\n--- Summary Metrics ---")
    print(df[["Baseline_CER", "Proposed_CER", "Baseline_WER", "Proposed_WER", "Baseline_F1", "Proposed_F1"]].mean())

if __name__ == "__main__":
    run_full_evaluation_pipeline()