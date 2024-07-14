import numpy as np
from typing import Union

RGB_COEFFICIENTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
BGR_COEFFICIENTS = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)

def is_grayscale(image: np.ndarray) -> bool:
    return image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1)

def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.0031308, 12.92 * image, 1.055 * (image ** (1 / 2.4)) - 0.055)

def to_grayscale(rgb_image: np.ndarray, channel_order: str = 'RGB') -> np.ndarray:
    assert 2 <= rgb_image.ndim <= 3 and rgb_image.shape[-1] == 3, "Invalid input image shape, expected (H, W, 3)"
    assert rgb_image.min() >= 0 and rgb_image.max() <= 255, "Input image values must be in the range [0, 255]"

    if is_grayscale(rgb_image):
        return rgb_image.reshape(*rgb_image.shape[:2], 1)
    
    image_float = rgb_image.astype(np.float32) / 255.0
    image_linear = srgb_to_linear(image_float)

    if channel_order == 'RGB':
        coefficients = RGB_COEFFICIENTS
    elif channel_order == 'BGR':
        coefficients = BGR_COEFFICIENTS
    else:
        raise ValueError("Invalid channel order, must be either 'RGB' or 'BGR'")

    gray_channel = image_linear @ coefficients
    gray_srgb = linear_to_srgb(gray_channel)

    if np.issubdtype(rgb_image.dtype, np.floating):
        grayscale_image = (gray_srgb * 255.).astype(rgb_image.dtype)
    else:
        grayscale_image = np.round(gray_srgb * 255.).astype(rgb_image.dtype)

    return grayscale_image
