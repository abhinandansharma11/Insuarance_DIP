import re
import pytesseract

class OCRExtractor:
    def __init__(self):
        import sys
        if sys.platform.startswith('win'):
            # Works on your local Windows laptop
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        else:
            # Works normally on the Streamlit Cloud public link
            pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    def run_baseline_ocr(self, img):
        return pytesseract.image_to_string(img)

    def run_enhanced_ocr(self, img):
        # Use Page Segmentation Mode 4 (Assume a single column of text of variable sizes)
        custom_config = r'--oem 3 --psm 4'
        return pytesseract.image_to_string(img, config=custom_config)

    def extract_key_fields(self, ocr_text):
        fields = {
            "policy_number": "NOT_FOUND",
            "claim_amount": "NOT_FOUND",
            "date": "NOT_FOUND",
            "document_type": "UNKNOWN"
        }
        
        if not ocr_text: return fields

        # 1. Match complex Policy Numbers (e.g. SLI/HLT/2026/014587)
        pol_match = re.search(r'[A-Z0-9]{3}/[A-Z0-9]{3}/\d{4}/\d{5,7}', ocr_text)
        if pol_match:
            fields["policy_number"] = pol_match.group(0)
        else:
            # Fallback for other formats
            pol_match_alt = re.search(r'(?:POL-|INS-)\d+', ocr_text)
            if pol_match_alt: fields["policy_number"] = pol_match_alt.group(0)

        # 2. Match Amount (Finds numbers like 5,00,000 or 11,130 and picks the largest)
        amounts = re.findall(r'(?<!\.)\b\d{1,3}(?:,\d{2,3})+\b(?!\.)', ocr_text)
        if amounts:
            # Sorts dynamically to grab the 'Sum Insured' instead of a small premium
            fields["claim_amount"] = max(amounts, key=lambda x: int(x.replace(',', '')))

        # 3. Match Date (e.g. 12 Nov 2026)
        date_match = re.search(r'\d{2}\s+[A-Za-z]{3}\s+\d{4}', ocr_text)
        if date_match:
            fields["date"] = date_match.group(0)

        # 4. Document Type
        text_lower = ocr_text.lower()
        if "health insurance" in text_lower or "policy certificate" in text_lower:
            fields["document_type"] = "policy_certificate"
        elif "claim" in text_lower:
            fields["document_type"] = "claim_form"
        else:
            fields["document_type"] = "general_document"

        return fields