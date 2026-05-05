import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _prepare_for_inference(img: Image.Image) -> Image.Image:
    """
    Normalize real-world inputs (screenshots, camera, PDF crops) to the
    clean white-background style the model was trained on.

    Steps applied in order:
      1. Convert to grayscale luminance to check background tone
      2. If dark background (mean < 0.45), invert — handles dark mode / night mode
      3. Auto-contrast to stretch histogram — fixes low-contrast scans/photos
      4. Mild sharpening to counter screenshot JPEG blur
    """
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    if arr.mean() < 0.45:
        img = ImageOps.invert(img.convert("RGB"))
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img.convert("RGB")


class Nav2TexImageProcessor(BaseImageProcessor):
    model_type = "nav2tex"

    def __init__(
        self,
        image_height=64,
        max_image_width=1024,
        patch_size=16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.patch_size = patch_size

    def preprocess(self, images, do_prepare=True, **kwargs) -> BatchFeature:
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")

            if do_prepare:
                img = _prepare_for_inference(img)

            w, h = img.size
            new_w = int(round(w * self.image_height / max(h, 1)))
            new_w = min(new_w, self.max_image_width)
            new_w = max((new_w // self.patch_size) * self.patch_size, self.patch_size)

            if (w, h) != (new_w, self.image_height):
                img = img.resize((new_w, self.image_height), Image.BILINEAR)

            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = (img_array - 0.5) / 0.5
            img_array = np.transpose(img_array, (2, 0, 1))
            processed_images.append(img_array)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type="pt")