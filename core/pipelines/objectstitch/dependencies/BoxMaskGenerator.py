from typing import Tuple
from PIL import Image, ImageDraw

class BoxMaskGenerator:
    def generate(
        self,
        resolution: Tuple[int, int],
        center: Tuple[int, int],
        size: Tuple[int, int]
    ) -> Tuple[Image.Image, Tuple[int, int, int, int], Tuple[int, int]]:
        # Crear una imagen completamente negra con la resolución especificada
        mask = Image.new("L", resolution, 0)
        draw = ImageDraw.Draw(mask)

        # Calcular las coordenadas del rectángulo blanco
        center_x, center_y = center
        width, height = size
        top_left_x = center_x - width // 2
        top_left_y = center_y - height // 2
        bottom_right_x = center_x + width // 2
        bottom_right_y = center_y + height // 2

        # Dibujar el rectángulo blanco en la imagen negra
        draw.rectangle((top_left_x, top_left_y, bottom_right_x, bottom_right_y), fill=255)

        # Definir la caja delimitadora y la forma (ancho, alto)
        bounding_box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        shape = (width, height)

        # Devolver la imagen, el bounding box y la forma
        return mask, bounding_box, shape
