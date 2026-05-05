import sys
import torch
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download


class Nav2TexPipeline:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        path = Path(repo_id_or_path)
        if not path.exists():
            path = Path(snapshot_download(repo_id_or_path))

        sys.path.insert(0, str(path))

        from nav2tex.tokenization_latex_ocr import LaTeXTokenizer
        from nav2tex.image_processing_latex_ocr import Nav2TexImageProcessor
        from nav2tex.processing_latex_ocr import Nav2TexProcessor
        from nav2tex.modeling_latex_ocr import Nav2TexModel
        from nav2tex.configuration_latex_ocr import Nav2TexConfig

        config = Nav2TexConfig.from_pretrained(str(path))
        image_processor = Nav2TexImageProcessor.from_pretrained(str(path))
        tokenizer = LaTeXTokenizer(str(path / "tokenizer.json"))
        processor = Nav2TexProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = Nav2TexModel.from_pretrained(str(path), config=config).to(device).eval()

        return cls(model=model, processor=processor, device=device)

    def __call__(self, image, max_new_tokens: int = None, num_beams: int = None):
        single = not isinstance(image, list)
        images = [image] if single else image

        loaded = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            loaded.append(img)

        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if num_beams is not None:
            kwargs["num_beams"] = num_beams

        # image processor handles variable-width images one at a time;
        # collect pixel_values as a list for NaViT's batched_images path
        pixel_values = [
            self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
            for img in loaded
        ]

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, **kwargs)

        results = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return results[0] if single else results