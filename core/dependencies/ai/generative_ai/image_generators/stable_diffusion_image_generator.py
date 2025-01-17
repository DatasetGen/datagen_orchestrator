from typing import Optional

from PIL import Image
import requests
from io import BytesIO
from enum import Enum, IntEnum

from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator


class StableDiffusionImageGenerator(ImageGenerator):
    class Models(Enum):
        SD3 = "sd3"
        CORE = "core"
        ULTRA = 'ultra'

    prompt : str
    api_key: str
    negative_prompt : Optional[str] = ""
    sd_model : Optional[Models] = Models.SD3
    aspect_ratio : Optional[str] = "1:1"

    def generate(self) -> Image.Image:
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            },
            files={"none": ''},
            data={
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "output_format": "png",
                "aspect_ratio": self.aspect_ratio
            },
        )

        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(str(response.json()))