from typing import Tuple
from PIL import Image
from pipelines.dependencies.image_cropper import ImageCropper
from pipelines.dependencies.image_paster import ImagePaster
from pipelines.dependencies.gen_ai.object_stitchers import LibcomObjectStitcher
from pipelines.dependencies.gen_ai.object_stitchers import ObjectStitcher
from pipelines.harmonization.dependencies.image_compositor import ImageCompositor
from pipelines.harmonization.dependencies.transparent_image_adjuster import TransparentImageAdjuster
from pipelines.objectstitch.dependencies.BoxMaskGenerator import BoxMaskGenerator
from pipelines.utils import plot_images, draw_square_inside_image
import sys

class ObjectStitcherDatasetGenerator:
    def __init__(self,
                 image_cropper : ImageCropper,
                 image_paster : ImagePaster,
                 image_compositor : ImageCompositor,
                 image_shape_adjuster : TransparentImageAdjuster,
                 box_mask_generator : BoxMaskGenerator,
                 object_stitcher : ObjectStitcher
                 ):
        self.image_cropper = image_cropper
        self.image_paster = image_paster
        self.image_compositor = image_compositor
        self.image_shape_adjuster = image_shape_adjuster
        self.mask_generator = box_mask_generator
        self.object_stitcher = object_stitcher


    def generate(self,  image : Image.Image, resolution: Tuple[int, int], save_as="result1"):
        point_of_crop = (450, 450)
        boat = Image.open("assets/boats/cargo_ship.png")
        cropped_image = self.image_cropper.crop(
            image=image,
            center=point_of_crop,
            resolution=resolution)

        mask, bbox, shape = self.mask_generator.generate(cropped_image.size, (cropped_image.size[0]//2, cropped_image.size[1]//2), (350, 270))

        stitched_image = self.object_stitcher.stitch(
            bg=cropped_image,
            fg_list=[boat,boat2, boat3],
            fg_mask_list=[mask,],
            bbox=bbox
        )

        pasted = self.image_paster.paste(
            original_image=image,
            pasted_image=stitched_image,
            center=point_of_crop
        )


        plot_images(
            [
                image,
                draw_square_inside_image(image, cropped_image.size, point_of_crop, border_width=7, center_radius=10),
                cropped_image,
                mask,
                draw_square_inside_image(cropped_image, shape, (cropped_image.size[0]//2, cropped_image.size[1]//2), border_width=4, center_radius=5),
                boat,
                stitched_image,
                draw_square_inside_image(pasted, (350, 270), point_of_crop, border_width=7, center_radius=10),
            ],
            ["Imagen original", "Posición de recorte", "Recorte", "Máscara de snittching", "Bounding box", "Imágen foreground" ,"Imágen con stitched",    "Imágen original con región copiada"],
            main_title="Pipeline",
            save_as=save_as
        )
        return pasted




folder = sys.argv[0] if sys.argv[0] else 0
image = Image.open("assets/bgs/bg.jpg")

for iteration in range(0, 1):
    dataset_generator = ObjectStitcherDatasetGenerator(
        image_cropper=ImageCropper(),
        image_paster=ImagePaster(),
        image_compositor=ImageCompositor(),
        image_shape_adjuster=TransparentImageAdjuster(),
        box_mask_generator=BoxMaskGenerator(),
        object_stitcher=LibcomObjectStitcher()
    )
    dataset_generator.generate(
        image=image,
        resolution=(512, 512),
        save_as="result.png"
        )