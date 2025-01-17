from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from typing import Tuple

def plot_images(images, titles, main_title="Images", save_as="result.png"):
    num_images = len(images)
    plt.figure(figsize=(5 * num_images, 5))
    plt.suptitle(main_title, fontsize=16)

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, num_images, i + 1)  # 1 row, num_images columns, ith subplot
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.show()
    plt.savefig(save_as)

def draw_square_inside_image(
        image: Image.Image,
        square_size: Tuple[int, int],
        center_point: Tuple[int, int],
        border_width: int = 3,
        center_radius: int = 0,
        color="red",
) -> Image.Image:
    image_with_square = image.copy()
    draw = ImageDraw.Draw(image_with_square)
    half_width, half_height = square_size[0] // 2, square_size[1] // 2
    top_left = (center_point[0] - half_width, center_point[1] - half_height)
    bottom_right = (center_point[0] + half_width, center_point[1] + half_height)
    for i in range(border_width):
        draw.rectangle(
            [
                (top_left[0] - i, top_left[1] - i),
                (bottom_right[0] + i, bottom_right[1] + i)
            ],
            outline=color
        )

    if center_radius > 0:
        draw.ellipse(
            [
                (center_point[0] - center_radius, center_point[1] - center_radius),
                (center_point[0] + center_radius, center_point[1] + center_radius)
            ],
            fill=color
        )

    return image_with_square
