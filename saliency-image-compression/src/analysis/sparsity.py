import numpy as np

from src.dct import block_processing, dct2
from src.quantization import quantize, FINE_Q, COARSE_Q
from src.saliency import compute_saliency


def quantized_coeffs_baseline(img_padded, block_size=8):
    """Baseline: every block uses FINE_Q."""
    blocks = block_processing(img_padded.astype(np.float32), block_size)  # (H/8, W/8, 8, 8)
    Hn, Wn = blocks.shape[0], blocks.shape[1]

    q = np.zeros((Hn, Wn, block_size, block_size), dtype=np.int32)
    for i in range(Hn):
        for j in range(Wn):
            d = dct2(blocks[i, j])
            q[i, j] = quantize(d, FINE_Q)
    return q


def quantized_coeffs_adaptive(img_padded, threshold=0.35, block_size=8):
    """Adaptive: saliency decides FINE_Q vs COARSE_Q per block."""
    img_f = img_padded.astype(np.float32)
    h, w = img_f.shape

    sal = compute_saliency(img_f)

    # mean saliency per 8x8 block
    sal_blocks = sal.reshape(h//block_size, block_size, w//block_size, block_size).swapaxes(1, 2)
    sal_vals = sal_blocks.mean(axis=(2, 3))  # (H/8, W/8)

    blocks = block_processing(img_f, block_size)
    Hn, Wn = blocks.shape[0], blocks.shape[1]

    q = np.zeros((Hn, Wn, block_size, block_size), dtype=np.int32)
    for i in range(Hn):
        for j in range(Wn):
            qtable = FINE_Q if sal_vals[i, j] >= threshold else COARSE_Q
            d = dct2(blocks[i, j])
            q[i, j] = quantize(d, qtable)
    return q


def zero_ratio(qcoeffs, exclude_dc=False):
    """Percent of zeros in quantized DCT coefficients."""
    arr = qcoeffs.reshape(-1, qcoeffs.shape[-2], qcoeffs.shape[-1])  # (num_blocks, 8, 8)
    if exclude_dc:
        arr = arr.copy()
        arr[:, 0, 0] = 1  # make DC non-zero so it doesn't count
    zeros = np.sum(arr == 0)
    total = arr.size
    return zeros / total

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
