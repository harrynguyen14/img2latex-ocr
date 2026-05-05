from transformers import ProcessorMixin
from nav2tex.image_processing_latex_ocr import Nav2TexImageProcessor
from nav2tex.tokenization_latex_ocr import LaTeXTokenizer

class Nav2TexProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        if images is None and text is None:
            raise ValueError("You must specify either images or text.")

        output = {}
        if images is not None:
            image_inputs = self.image_processor(images, return_tensors=return_tensors, **kwargs)
            output.update(image_inputs)

        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            output.update(text_inputs)

        return output

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)