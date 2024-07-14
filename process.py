import random
from PIL import Image
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from image_pixelation import prepare_pixelated_image
from color_conversion import to_grayscale


class RandomImagePixelationDataset(Dataset):
    def __init__(self, image_dir: str, width_range: tuple, height_range: tuple, size_range: tuple, dtype=None):
        self._check_range(width_range, "width_range")
        self._check_range(height_range, "height_range")
        self._check_range(size_range, "size_range")

        self.image_files = glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
        random.shuffle(self.image_files)

        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype

    def _check_range(self, value_range: tuple, name: str):
        min_val, max_val = value_range
        if min_val < 2:
            raise ValueError(f"Minimum value of {name} must be at least 2. Got {min_val}")
        if min_val > max_val:
            raise ValueError(
                f"Minimum value of {name} must not be greater than maximum value. Got {min_val} > {max_val}")

    def __getitem__(self, index: int):
        try:
            image_file = self.image_files[index]
            image = Image.open(image_file)
            image_array = np.array(image, dtype=self.dtype)
            image_array = self.to_grayscale(image_array)

            rng = random.Random(index)
            width = rng.randint(*self.width_range)
            height = rng.randint(*self.height_range)
            size = rng.randint(*self.size_range)

            width = min(width, image_array.shape[1])
            height = min(height, image_array.shape[0])

            x = rng.randint(0, image_array.shape[1] - width)
            y = rng.randint(0, image_array.shape[0] - height)

            pixelated_image, known_array, target_array = self.prepare_pixelated_image(
                image_array, x, y, width, height, size
            )

            return pixelated_image, known_array, target_array, image_file
        except (IOError, ValueError) as e:
            # Handle specific exceptions here
            print(f"Error processing image at index {index}: {e}")
            return None

    def __len__(self):
        return len(self.image_files)
