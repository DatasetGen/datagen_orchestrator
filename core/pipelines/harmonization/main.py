import uuid
from typing import Tuple, List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os

from core.dependencies.ai.discriminative_ai.background_removers.background_remover import BackgroundRemover
from core.dependencies.ai.discriminative_ai.background_removers.mock_background_remover import MockBackgroundRemover
from core.dependencies.ai.discriminative_ai.background_removers.stable_diffusion_background_remover import \
    StableDiffusionBackgroundRemover
from core.dependencies.ai.discriminative_ai.point_extractors.mmseg_point_extractor import MMSegPointExtractor
from core.dependencies.ai.discriminative_ai.point_extractors.point_extractor import PointExtractor
from core.dependencies.ai.generative_ai.image_generators.mock_image_generator import MockImageGenerator
from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator
from core.dependencies.ai.generative_ai.image_generators.stable_diffusion_image_generator import \
    StableDiffusionImageGenerator
from core.dependencies.ai.generative_ai.image_generators.sthocastic_image_generator import StochasticImageGenerator
from core.dependencies.ai.generative_ai.image_harmonizers.image_harmonizer import ImageHarmonizer
from core.dependencies.ai.generative_ai.image_harmonizers.mock_image_harnonizer import MockImageHarmonizer
from core.dependencies.ai.generative_ai.image_inpainters.image_inpainter import ImageInpainter
from core.dependencies.ai.generative_ai.image_inpainters.stable_diffusion_image_inpainter import \
    StableDiffusionImageInpainter
from core.dependencies.api.mmseg_api import MMSegAPI
from core.dependencies.utils.datasets.bounding_box_generator import BoundingBoxGenerator
from core.dependencies.utils.image.image_editors.image_paster import ImagePaster
from core.dependencies.utils.image.mask_generators.alpha_mask_generator import AlphaMaskGenerator
from core.pipelines.pipeline import DatasetGenerationPipeline, BoundingBox
from core.utils import plot_images


class HarmonizationDatasetGenerator(DatasetGenerationPipeline):
    n_boats : int = 1
    boat_size : int = 500
    crop_resolution : Tuple[int, int] = (512, 512)
    boat_generator : ImageGenerator
    background_generator : ImageGenerator
    background_remover : BackgroundRemover
    point_extractor : PointExtractor
    harmonizer : ImageHarmonizer
    inpainter: ImageInpainter

    def generate(self) -> Tuple[Image.Image, List[BoundingBox]]:
        bg = self.background_generator.generate()
        points = self.point_extractor.extract(bg)
        boats_with_transparency = Image.new("RGBA", bg.size, (0, 0, 0, 0))
        bboxs = []
        gen_boats = []

        for x in points:
            single_boat = Image.new("RGBA", bg.size, (0, 0, 0, 0))
            base_boat = self.boat_generator.generate()
            gen_boats.append(base_boat)
            boat = self.background_remover\
                .remove(base_boat)\
                .resize((self.boat_size, ((base_boat.size[1] * self.boat_size) // base_boat.size[0]))).convert("RGBA")
            single_boat = ImagePaster(center=x, patch=boat).edit(single_boat)
            boats_with_transparency = ImagePaster(center=x, patch=boat).edit(boats_with_transparency)
            bboxs.append(BoundingBoxGenerator().generate(alpha_img=single_boat))

        bg = ImagePaster(center=(bg.size[0]//2, bg.size[1]//2), patch=boats_with_transparency).edit(bg)
        outside_inpainting_mask = AlphaMaskGenerator(
            alpha_image=boats_with_transparency,
            type=AlphaMaskGenerator.Type.border_outside,
            border_width=10,
            strength=255
        ).generate()
        inside_inpainting_mask = AlphaMaskGenerator(
            alpha_image=boats_with_transparency,
            type=AlphaMaskGenerator.Type.inside,
            strength=100,
            border_width=3
        ).generate()

        inpainted_outside = self.inpainter.inpaint(original_image=bg, mask_image=outside_inpainting_mask)
        inpainted_inside = self.inpainter.inpaint(original_image=inpainted_outside, mask_image=inside_inpainting_mask)
        plot_images([
            *gen_boats,
            boats_with_transparency,
            bg,
            outside_inpainting_mask,
            inpainted_outside,
            inside_inpainting_mask,
            inpainted_inside
        ],
        [
            *["boats" for _ in gen_boats],
            "boats with transparency",
            "background",
            "outside_inpainting_mask",
            "outside inpainting",
            "inside inpainting mask",
            "inside inpainting"
        ])
        return inpainted_inside, bboxs



"""
load_dotenv()
img, bbox = HarmonizationDatasetGenerator(
    boat_generator=StableDiffusionImageGenerator(api_key=os.getenv("SD_API_KEY", ""), prompt="A photo of a fishing ship with white background"),
    background_generator=MockImageGenerator(route="../../../data_assets/bgs/fondo8.jpg"),
    background_remover=StableDiffusionBackgroundRemover(api_key=os.getenv("SD_API_KEY", "")),
    point_extractor=MMSegPointExtractor(api=MMSegAPI(url=os.getenv("MMSEG_API_URL", "http://100.103.218.9:4553/v1"))),
    harmonizer=MockImageHarmonizer(),
    inpainter=StableDiffusionImageInpainter(api_key=os.getenv("SD_API_KEY", ""), prompt="A coastal view of the sea and a background of the city", negative_prompt="boats")
).generate()
plt.imshow(img)
plt.show()
img.save("harmonization.png")
"""

load_dotenv()
img, bboxs = HarmonizationDatasetGenerator(
    boat_size=200,
    crop_resolution=(512, 512),
    boat_generator=StochasticImageGenerator(dir="../../../data_assets/boats/"),
    #boat_generator=StableDiffusionImageGenerator(api_key=os.getenv("SD_API_KEY", ""), prompt="motor fishing boat in a white background", negative_prompt="background, beach, sea, ocean, mountains"),
    #boat_generator=MockImageGenerator(route="../../../data_assets/boats/without_bg/image_2-removebg-preview.png"),
    background_generator=MockImageGenerator(route="../../../data_assets/bgs/fondo10.jpg"),
    #background_remover=StableDiffusionBackgroundRemover(api_key=os.getenv("SD_API_KEY", "")),
    background_remover=MockBackgroundRemover(),
    point_extractor=MMSegPointExtractor(api=MMSegAPI(url=os.getenv("MMSEG_API_URL", "http://100.103.218.9:4553/v1")), n_points=2),
    harmonizer=MockImageHarmonizer(),
    inpainter=StableDiffusionImageInpainter(api_key=os.getenv("SD_API_KEY", ""), prompt="A coastal view of open sea with two boats", negative_prompt="boats")
    #inpainter=MockImageInpainter()
).generate()
#
img.save(uuid.uuid4().hex + ".png")
[BoundingBoxGenerator().paint(img, bbox, color=(255, 0, 0)) for bbox in bboxs]
plt.imshow(img)
plt.show()
img.save(uuid.uuid4().hex + ".png")
