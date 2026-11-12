import cv2
import numpy as np
from typing import Tuple

class FrequencyDomainFilter:
    """
    Implements frequency-domain filtering techniques: Fast Fourier Transform (FFT),
    Butterworth/Gaussian low-pass filters, Wiener deconvolution for blur restoration,
    and homomorphic filtering for uneven illumination removal.
    """

    def estimate_blur_variance(self, image: np.ndarray) -> float:
        """Estimates image blur using the variance of the Laplacian operator."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def apply_lowpass_filter(self, image: np.ndarray, cutoff: int = 30) -> np.ndarray:
        """Applies an FFT-based Butterworth low-pass filter to remove high-frequency noise."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. FFT
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # 2. Butterworth Lowpass Mask
        mask = np.zeros((rows, cols), np.float32)
        n = 2  # filter order
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - crow)**2 + (j - ccol)**2)
                mask[i, j] = 1.0 / (1.0 + (d / cutoff)**(2 * n))

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.clip(img_back, 0, 255).astype(np.uint8)

    def apply_wiener_deconvolution(self, image: np.ndarray, kernel_size: int = 5, K: float = 0.01) -> np.ndarray:
        """Restores motion-blurred text using estimated Point Spread Function (PSF) and Wiener filtering."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Create a motion blur PSF approximation
        psf = np.zeros((kernel_size, kernel_size))
        psf[kernel_size // 2, :] = 1.0 / kernel_size

        # Pad PSF to image size
        img_float = gray.astype(np.float32) / 255.0
        psf_padded = np.zeros_like(img_float)
        kh, kw = psf.shape
        psf_padded[:kh, :kw] = psf
        
        # Circular shift PSF center
        psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
        psf_padded = np.roll(psf_padded, -kw // 2, axis=1)

        # Wiener filter formula in frequency domain
        H = np.fft.fft2(psf_padded)
        G = np.fft.fft2(img_float)
        
        H_conj = np.conj(H)
        denom = H_conj * H + K
        F_hat = (H_conj * G) / denom
        
        restored = np.abs(np.fft.ifft2(F_hat))
        restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
        return restored

    def apply_homomorphic_filter(self, image: np.ndarray, d0: int = 30, gamma_l: float = 0.5, gamma_h: float = 1.5) -> np.ndarray:
        """Removes uneven illumination and harsh shadows using log-domain high-emphasis filtering."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_float = gray.astype(np.float32) + 1.0
        
        # 1. Log domain transformation
        img_log = np.log(gray_float)
        
        # 2. FFT
        dft = np.fft.fft2(img_log)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # 3. Homomorphic filter mask (Gaussian high-emphasis)
        u = np.arange(rows)
        v = np.arange(cols)
        U, V = np.meshgrid(v - ccol, u - crow)
        D = np.sqrt(U**2 + V**2)
        
        H = (gamma_h - gamma_l) * (1 - np.exp(-(D**2) / (2 * (d0**2)))) + gamma_l
        
        # Apply filter
        filtered_fft = dft_shift * H
        f_ishift = np.fft.ifftshift(filtered_fft)
        img_back = np.fft.ifft2(f_ishift)
        
        # 4. Exponential domain recovery
        img_exp = np.exp(np.real(img_back)) - 1.0
        img_normalized = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
        
        return img_normalized.astype(np.uint8)