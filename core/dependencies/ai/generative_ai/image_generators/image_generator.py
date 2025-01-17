from abc import abstractmethod
from PIL import Image
from pydantic import BaseModel


class ImageGenerator(BaseModel):
    @abstractmethod
    def generate(self) -> Image.Image:
        ...


