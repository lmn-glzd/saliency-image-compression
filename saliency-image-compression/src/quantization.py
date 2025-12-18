# src/quantization.py
import numpy as np

# Standard luminance quantization table from JPEG
STANDARD_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
])

COARSE_Q = STANDARD_Q * 4  # more compression
FINE_Q   = STANDARD_Q      # less compression (higher quality)

def quantize(block_dct: np.ndarray, q_table: np.ndarray):
    """
    Quantize DCT block using q_table.
    """
    return (block_dct / q_table).round().astype(int)

def dequantize(block_q: np.ndarray, q_table: np.ndarray):
    """
    Dequantize quantized block.
    """
    return (block_q * q_table).astype(int)
