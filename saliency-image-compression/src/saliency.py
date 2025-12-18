# src/saliency.py
import cv2
import numpy as np

def compute_saliency(image: np.ndarray):
    """
    Compute a saliency map for the given image (grayscale or color).
    Returns a float map normalized to [0,1].
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, sal_map) = saliency.computeSaliency(image)
    sal_map = cv2.normalize(sal_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return sal_map
