# src/analysis/frequency.py

import numpy as np

from src.dct import block_processing, dct2
from src.quantization import quantize, FINE_Q, COARSE_Q
from src.saliency import compute_saliency
from src.analysis.sparsity import subband_zero_ratio


def get_quantized_dct_blocks(img, mode="baseline", threshold=0.35, block_size=8):
    """
    Compute quantized DCT coefficient blocks under different compression modes.

    Supported modes:
    - "baseline": standard JPEG-style compression using a fixed quantization table
    - "adaptive": saliency-based adaptive quantization (block-level)
    - "adaptive_freq": saliency + frequency-aware adaptive quantization
                       (different quantization strength per frequency band)

    This function operates purely in the frequency domain and is intended
    for analysis (e.g., sparsity / subband statistics), not image reconstruction.

    Parameters:
        img (ndarray): Padded grayscale image (H x W)
        mode (str): Compression mode ("baseline", "adaptive", "adaptive_freq")
        threshold (float): Saliency threshold for adaptive modes
        block_size (int): DCT block size (default: 8)

    Returns:
        ndarray: Quantized DCT blocks with shape
                 (H/block_size, W/block_size, block_size, block_size)
    """
    h, w = img.shape

    # Split image into non-overlapping spatial blocks
    blocks = block_processing(img, block_size)

    # Saliency map is only required for adaptive modes
    if mode != "baseline":
        sal_map = compute_saliency(img)

        # Convert pixel-level saliency map to block-level saliency values
        sal_blocks = sal_map.reshape(
            h // block_size, block_size,
            w // block_size, block_size
        ).swapaxes(1, 2)

        # Mean saliency per block (H/8 x W/8)
        sal_vals = sal_blocks.mean(axis=(2, 3))

    # Container for quantized DCT coefficients
    qd_blocks = np.zeros_like(blocks)

    # Processing of each block independently
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):

            # Compute DCT of current spatial block
            d = dct2(blocks[i, j])

            if mode == "baseline":
                # Uniform quantization for all blocks 
                q = FINE_Q
                qd_blocks[i, j] = quantize(d, q)

            elif mode == "adaptive":
                # Block-level saliency-based quantization
                q = FINE_Q if sal_vals[i, j] > threshold else COARSE_Q
                qd_blocks[i, j] = quantize(d, q)

            elif mode == "adaptive_freq":
                # Frequency-aware adaptive quantization:
                # lower frequencies preserved more carefully,
                # higher frequencies quantized more aggressively
                qd = np.zeros_like(d)

                for u in range(block_size):
                    for v in range(block_size):

                        if sal_vals[i, j] > threshold:
                            # Salient block: preserve low frequencies
                            if u + v <= 4:
                                q = FINE_Q
                            elif u + v <= 7:
                                q = (FINE_Q + COARSE_Q) // 2
                            else:
                                q = COARSE_Q
                        else:
                            # Non-salient block: aggressive quantization
                            q = COARSE_Q

                        qd[u, v] = quantize(d[u, v], q[u, v])

                qd_blocks[i, j] = qd

    return qd_blocks


def analyze_subbands_qd(qd_blocks):
    """
    Analyze sparsity (zero-ratio) in different frequency subbands
    of quantized DCT blocks.

    Subbands:
    - low:    low-frequency coefficients
    - mid:    mid-frequency coefficients
    - high:   high-frequency coefficients

    Parameters:
        qd_blocks (ndarray): Quantized DCT blocks

    Returns:
        dict: Mean zero-ratio per subband across all blocks
    """
    stats = {"low": [], "mid": [], "high": []}

    # Iterate over all blocks
    for i in range(qd_blocks.shape[0]):
        for j in range(qd_blocks.shape[1]):

            # Compute zero-ratio for each subband of the block
            z = subband_zero_ratio(qd_blocks[i, j])

            for k in stats:
                stats[k].append(z[k])

    # Average statistics across all blocks
    return {k: np.mean(stats[k]) for k in stats}
