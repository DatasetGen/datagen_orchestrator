from PIL import Image

from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator


class FluxImageGenerator(ImageGenerator):
    def generate(self) -> Image.Image:
        pass