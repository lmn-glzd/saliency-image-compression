# src/adaptive_compression.py
import numpy as np
import cv2
from .dct import block_processing, dct2, idct2, reconstruct_image
from .quantization import quantize, dequantize, COARSE_Q, FINE_Q
from .saliency import compute_saliency
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def select_frequency_qtable(i, j, saliency_value, threshold):
    """
    Decide quantization strength based on saliency + frequency location.
    (i, j) are DCT coefficient indices.
    """
    # salient region → preserve low frequencies
    if saliency_value > threshold:
        if i + j <= 4:      # very low frequency
            return FINE_Q
        elif i + j <= 7:    # mid frequency
            return (FINE_Q + COARSE_Q) // 2
        else:               # high frequency
            return COARSE_Q
    else:
        # non-salient → compress aggressively
        return COARSE_Q


def baseline_compress(image, block_size=8):
    img = image.copy()

    blocks = block_processing(img, block_size)

    dct_blocks = np.array([[dct2(b) for b in row] for row in blocks])
    q_blocks  = np.array([[quantize(b, FINE_Q) for b in row] for row in dct_blocks])
    dq_blocks = np.array([[dequantize(b, FINE_Q) for b in row] for row in q_blocks])
    recon_blocks = np.array([[idct2(b) for b in row] for row in dq_blocks])

    recon_img = reconstruct_image(recon_blocks, block_size)
    return np.clip(recon_img, 0, 255).astype(np.uint8)


def adaptive_compress(image: np.ndarray, threshold: float = 0.35):
    img = image.copy().astype(np.float32)
    block_size = 8

    h, w = img.shape

    # 1) Compute saliency map
    sal_map = compute_saliency(img)

    # 2) Convert saliency to block-level map (H/8 x W/8)
    sal_blocks = sal_map.reshape(h//block_size, block_size,
                                 w//block_size, block_size).swapaxes(1, 2)
    sal_vals = sal_blocks.mean(axis=(2, 3))  # shape: (H/8, W/8)

    # 3) Split image into blocks (H/8, W/8, 8,8)
    blocks = block_processing(img, block_size)

    # new empty array for reconstructed blocks
    out_blocks = np.zeros_like(blocks)

    # 4) Apply adaptive quantization block-by-block
    num_block_h = h // block_size
    num_block_w = w // block_size

    for i in range(num_block_h):
        for j in range(num_block_w):

            block = blocks[i, j]

            qtable = FINE_Q if sal_vals[i, j] > threshold else COARSE_Q

            d = dct2(block)
            qd = quantize(d, qtable)
            deq = dequantize(qd, qtable)
            out_blocks[i, j] = idct2(deq)

    # 5) Reconstruct full image
    recon = reconstruct_image(out_blocks, block_size)
    return np.clip(recon, 0, 255).astype(np.uint8)


def evaluate(original: np.ndarray, reconstructed: np.ndarray):
    p = psnr(original, reconstructed)
    s = ssim(original, reconstructed)
    return p, s

def adaptive_compress_freq(image: np.ndarray, threshold: float = 0.35):
    """
    Extension: Saliency + Frequency-aware adaptive compression.
    Main saliency-based method remains unchanged.
    """
    img = image.copy().astype(np.float32)
    block_size = 8
    h, w = img.shape

    sal_map = compute_saliency(img)

    sal_blocks = sal_map.reshape(
        h // block_size, block_size,
        w // block_size, block_size
    ).swapaxes(1, 2)
    sal_vals = sal_blocks.mean(axis=(2, 3))

    blocks = block_processing(img, block_size)
    out_blocks = np.zeros_like(blocks)

    for i in range(h // block_size):
        for j in range(w // block_size):

            block = blocks[i, j]
            d = dct2(block)
            qd = np.zeros_like(d)

            for u in range(block_size):
                for v in range(block_size):

                    # frequency-aware decision
                    if sal_vals[i, j] > threshold:
                        if u + v <= 4:
                            q = FINE_Q
                        elif u + v <= 7:
                            q = (FINE_Q + COARSE_Q) // 2
                        else:
                            q = COARSE_Q
                    else:
                        q = COARSE_Q

                    qd[u, v] = quantize(d[u, v], q[u, v])

            deq = dequantize(qd, COARSE_Q)
            out_blocks[i, j] = idct2(deq)

    recon = reconstruct_image(out_blocks, block_size)
    return np.clip(recon, 0, 255).astype(np.uint8)


def subband_zero_ratio(dct_block):
    """
    Compute zero ratio in low / mid / high frequency subbands.
    """
    low = dct_block[:3, :3]
    mid = dct_block[3:6, 3:6]
    high = dct_block[6:, 6:]

    return {
        "low":  np.mean(low == 0),
        "mid":  np.mean(mid == 0),
        "high": np.mean(high == 0),
    }
