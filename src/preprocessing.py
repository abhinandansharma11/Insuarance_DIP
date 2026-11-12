import cv2
import numpy as np
from typing import Tuple

class DocumentPreprocessor:
    """
    Implements core DIP preprocessing techniques including Canny-based document contour
    detection, perspective rectification (homography), gamma correction, and CLAHE.
    """

    def detect_document_contour(self, image: np.ndarray) -> np.ndarray:
        """Detects the 4 corner points of the primary document quadrilateral using Canny edges."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]], dtype=np.float32)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx.reshape(4, 2)
                break

        return self._order_points(doc_contour)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Orders corner points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def correct_perspective(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Applies homography warp to flatten and rectify the tilted document."""
        (tl, tr, br, bl) = pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
        heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.float32([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ])

        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def adjust_gamma(self, image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """Applies non-linear gamma correction to normalize overall brightness."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def compare_histogram_equalization(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compares global histogram equalization against Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Global Hist Eq
        global_eq = cv2.equalizeHist(gray)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_out = clahe.apply(gray)

        return global_eq, clahe_out