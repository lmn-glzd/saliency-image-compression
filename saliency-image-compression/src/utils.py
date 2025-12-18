import numpy as np

def pad_to_block_size(img, block_size=8):
    """Pad image so that its dimensions become multiples of block_size."""
    h, w = img.shape
    new_h = ((h + block_size - 1) // block_size) * block_size
    new_w = ((w + block_size - 1) // block_size) * block_size

    padded = np.zeros((new_h, new_w), dtype=img.dtype)
    padded[:h, :w] = img
    return padded
