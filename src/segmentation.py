import cv2
import numpy as np

class DocumentSegmentation:
    def __init__(self):
        pass

    def apply_global_binarization(self, gray_img):
        # Otsu thresholding
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure background is white and text is black
        if np.sum(binary == 0) > np.sum(binary == 255):
            binary = cv2.bitwise_not(binary)
        return binary

    def apply_local_binarization(self, gray_img, method="otsu"):
        return self.apply_global_binarization(gray_img)

    def apply_morphology(self, binary_img, operation="closing"):
        # ROOT CAUSE FIX: OpenCV morphology assumes text is WHITE.
        # We MUST invert the image so text is white, apply morphology, then invert back.
        inv_binary = cv2.bitwise_not(binary_img)
        
        # Use a very small 2x2 kernel so thin fonts are not destroyed
        kernel = np.ones((2, 2), np.uint8)
        
        if operation == "closing":
            processed = cv2.morphologyEx(inv_binary, cv2.MORPH_CLOSE, kernel)
        else:
            processed = inv_binary
            
        # Invert back to Black Text on White Background for OCR
        return cv2.bitwise_not(processed)

    def filter_connected_components(self, binary_img):
        return binary_img