import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import Levenshtein
from typing import Dict, Any

class EvaluationMetrics:
    """
    Computes rigorous image quality metrics (PSNR, SSIM), text error rates
    (Character Error Rate, Word Error Rate), and regex entity extraction F1 scores.
    """

    def calculate_image_quality(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Calculates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM)."""
        # Ensure same shape and dtype
        if original.shape != processed.shape:
            processed = np.resize(processed, original.shape)

        score_psnr = float(psnr(original, processed, data_range=255))
        score_ssim = float(ssim(original, processed, data_range=255, channel_axis=2 if len(original.shape)==3 else None))

        return {
            "PSNR": score_psnr,
            "SSIM": score_ssim
        }

    def calculate_text_error_rates(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculates Character Error Rate (CER) and Word Error Rate (WER)."""
        ref_clean = reference.strip()
        hyp_clean = hypothesis.strip()

        # Character level distance (Levenshtein)
        char_dist = Levenshtein.distance(ref_clean, hyp_clean)
        char_len = max(1, len(ref_clean))
        cer = float(char_dist / char_len)

        # Word level distance
        ref_words = ref_clean.split()
        hyp_words = hyp_clean.split()
        
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion = d[i][j-1] + 1
                    deletion = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        word_dist = d[len(ref_words)][len(hyp_words)]
        word_len = max(1, len(ref_words))
        wer = float(word_dist / word_len)

        return {
            "CER": round(cer, 4),
            "WER": round(wer, 4)
        }

    def calculate_extraction_f1(self, true_fields: Dict[str, str], extracted_fields: Dict[str, str]) -> float:
        """Calculates macro F1-score across parsed key-value insurance fields."""
        matches = 0
        total = len(true_fields)

        for key, true_val in true_fields.items():
            ext_val = extracted_fields.get(key, "")
            if true_val.lower() == ext_val.lower() and true_val != "NOT_FOUND":
                matches += 1

        precision = matches / max(1, sum(1 for v in extracted_fields.values() if v != "NOT_FOUND"))
        recall = matches / max(1, total)

        if precision + recall == 0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))