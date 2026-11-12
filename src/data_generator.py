import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from typing import Dict, Any, Tuple

class InsuranceDataGenerator:
    """
    Generates synthetic insurance documents (claim forms, invoices, ID cards)
    complete with realistic degradations such as perspective tilt, noise, blur,
    shadow gradients, and compression artifacts. Exports JSON ground truth.
    """

    def __init__(self, output_dir: str = "data/synthetic_dataset"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.gt_dir = os.path.join(output_dir, "ground_truth")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.gt_dir, exist_ok=True)

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Attempts to load a standard truetype font, falls back to default if unavailable."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    def generate_synthetic_document(self, doc_type: str, index: int) -> Tuple[str, str, Dict[str, Any]]:
        """
        Creates a base PIL image of a fictional insurance document, applies degradations,
        saves the image and ground truth metadata.
        """
        width, height = 800, 1000
        base_img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(base_img)
        
        font_large = self._get_font(24)
        font_med = self._get_font(16)
        font_small = self._get_font(12)

        # Generate fictional data fields
        policy_no = f"POL-{random.randint(100000, 999999)}"
        claim_amt = f"${random.randint(500, 15000)}.00"
        name = random.choice(["Aarav Sharma", "Priya Patel", "Vikram Singh", "Ananya Iyer"])
        date = f"2026-10-{random.randint(1, 28):02d}"

        if doc_type == "claim_form":
            draw.text((50, 50), "MEDICAL INSURANCE CLAIM FORM", fill=(0, 0, 0), font=font_large)
            draw.text((50, 120), f"Policy Number: {policy_no}", fill=(0, 0, 0), font=font_med)
            draw.text((50, 160), f"Claimant Name: {name}", fill=(0, 0, 0), font=font_med)
            draw.text((50, 200), f"Incident Date: {date}", fill=(0, 0, 0), font=font_med)
            draw.text((50, 240), f"Requested Amount: {claim_amt}", fill=(0, 0, 0), font=font_med)
            draw.text((50, 300), "Declaration: I hereby declare that the information provided is true.", fill=(50, 50, 50), font=font_small)

        elif doc_type == "invoice":
            draw.text((50, 50), "HOSPITAL REPAIR & TREATMENT INVOICE", fill=(0, 0, 0), font=font_large)
            draw.text((50, 120), f"Invoice Ref: INV-{random.randint(1000, 9999)}", fill=(0, 0, 0), font=font_med)
            draw.text((50, 160), f"Associated Policy: {policy_no}", fill=(0, 0, 0), font=font_med)
            draw.text((50, 200), f"Patient: {name}", fill=(0, 0, 0), font=font_med)
            draw.text((50, 250), "Description                          Qty    Total", fill=(0, 0, 0), font=font_med)
            draw.text((50, 280), f"Emergency Care Consultation         1      {claim_amt}", fill=(0, 0, 0), font=font_med)

        elif doc_type == "id_card":
            draw.rectangle([50, 50, 600, 350], outline=(0, 0, 0), width=3)
            draw.text((80, 80), "NATIONAL HEALTH INSURANCE ID", fill=(0, 0, 0), font=font_large)
            draw.text((80, 150), f"Cardholder: {name}", fill=(0, 0, 0), font=font_med)
            draw.text((80, 190), f"Policy ID: {policy_no}", fill=(0, 0, 0), font=font_med)
            draw.text((80, 230), f"Valid Thru: 2030-12-31", fill=(0, 0, 0), font=font_med)

        # Apply degradations
        degraded_img = self._apply_degradations(base_img)

        # File paths
        filename = f"{doc_type}_{index:03d}.jpg"
        img_path = os.path.join(self.images_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename.replace(".jpg", ".json"))

        degraded_img.save(img_path, "JPEG", quality=random.randint(65, 90))

        ground_truth = {
            "document_type": doc_type,
            "policy_number": policy_no,
            "claim_amount": claim_amt,
            "name": name,
            "date": date,
            "image_path": img_path
        }

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=4)

        return img_path, gt_path, ground_truth

    def _apply_degradations(self, img: Image.Image) -> Image.Image:
        """Applies a pipeline of random geometric and photometric degradations."""
        np_img = np.array(img)

        # 1. Perspective Tilt
        if random.random() > 0.3:
            np_img = self.apply_perspective_warp(np_img)

        img = Image.fromarray(np_img)

        # 2. Gaussian Noise & Salt-and-Pepper
        if random.random() > 0.3:
            img = self.apply_blur_and_noise(img)

        # 3. Illumination shadows & contrast reduction
        if random.random() > 0.3:
            img = self.apply_illumination_artifacts(img)

        return img

    def apply_perspective_warp(self, np_img: np.ndarray) -> np.ndarray:
        """Simulates camera angle tilt using homography transformation."""
        import cv2
        h, w = np_img.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        offset = random.randint(30, 80)
        pts2 = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(np_img, matrix, (w, h), borderValue=(255, 255, 255))
        return warped

    def apply_blur_and_noise(self, img: Image.Image) -> Image.Image:
        """Adds motion blur or Gaussian noise."""
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.8)))
        
        # Add random salt and pepper noise
        np_img = np.array(img)
        row, col, ch = np_img.shape
        s_vs_p = 0.5
        amount = 0.002
        # Salt noise
        num_salt = np.ceil(amount * np_img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_img.shape[:2]]
        np_img[tuple(coords)] = 255

        # Pepper noise
        num_pepper = np.ceil(amount * np_img.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_img.shape[:2]]
        np_img[tuple(coords)] = 0

        return Image.fromarray(np_img)

    def apply_illumination_artifacts(self, img: Image.Image) -> Image.Image:
        """Simulates uneven lighting/shadow gradient across the document."""
        np_img = np.array(img).astype(np.float32)
        h, w = np_img.shape[:2]
        
        # Create a gradient shadow mask
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xv, yv = np.meshgrid(x, y)
        shadow = 1.0 - 0.4 * (xv * yv)
        
        for c in range(3):
            np_img[:, :, c] *= shadow
            
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)