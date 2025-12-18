# src/dct.py
import cv2
import numpy as np


def dct2(block):
    """
    Apply 2D Discrete Cosine Transform (DCT) to an image block.

    Parameters:
        block (ndarray): 2D image block (e.g., 8x8 pixels)

    Returns:
        ndarray: DCT coefficients of the block
    """
    block = np.float32(block)   # OpenCV DCT expects float32
    return cv2.dct(block)


def idct2(block):
    """
    Apply inverse 2D Discrete Cosine Transform (IDCT) to a DCT block.

    Parameters:
        block (ndarray): 2D array of DCT coefficients

    Returns:
        ndarray: Reconstructed image block in spatial domain
    """
    block = np.float32(block)
    return cv2.idct(block)


def block_processing(image, block_size):
    """
    Split an image into non-overlapping square blocks.

    This function assumes that the image dimensions are exact
    multiples of the block size (e.g., after padding).

    Parameters:
        image (ndarray): Grayscale image (H x W)
        block_size (int): Size of each square block (e.g., 8)

    Returns:
        ndarray: Array of shape (H/block_size, W/block_size, block_size, block_size)
    """
    h, w = image.shape
    assert h % block_size == 0 and w % block_size == 0, \
        "Image dimensions must be multiples of block_size"

    blocks = []

    # Iterate over image in steps of block_size
    for i in range(0, h, block_size):
        row = []
        for j in range(0, w, block_size):
            # Extract one block
            row.append(image[i:i + block_size, j:j + block_size])
        blocks.append(row)

    return np.array(blocks)


def reconstruct_image(blocks, block_size):
    """
    Reconstruct the full image from its block representation.

    Parameters:
        blocks (ndarray): Array of blocks with shape
                          (num_blocks_h, num_blocks_w, block_size, block_size)
        block_size (int): Size of each block

    Returns:
        ndarray: Reconstructed image in spatial domain
    """
    num_blocks_h, num_blocks_w = blocks.shape[:2]

    # Compute full image dimensions
    h = num_blocks_h * block_size
    w = num_blocks_w * block_size

    image = np.zeros((h, w), dtype=np.float32)

    # Place each block back into its spatial location
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            image[
                i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size
            ] = blocks[i, j]

    return image
