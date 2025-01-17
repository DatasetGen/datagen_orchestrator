from abc import abstractmethod
from typing import Tuple, List

from PIL import Image
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x : float
    y : float
    w : float
    h : float

class DatasetGenerationPipeline(BaseModel):
    @abstractmethod
    def generate(self) -> Tuple[Image.Image, List[BoundingBox]]:
        ...