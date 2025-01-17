import uuid
from typing import Tuple, List
from PIL import Image
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os

from core.dependencies.ai.discriminative_ai.background_removers.background_remover import BackgroundRemover
from core.dependencies.ai.discriminative_ai.background_removers.stable_diffusion_background_remover import \
    StableDiffusionBackgroundRemover
from core.dependencies.ai.discriminative_ai.point_extractors.point_extractor import PointExtractor
from core.dependencies.ai.discriminative_ai.point_extractors.sparse_point_extractor import SparsePointExtractor
from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator
from core.dependencies.ai.generative_ai.image_generators.stable_diffusion_image_generator import \
    StableDiffusionImageGenerator
from core.dependencies.ai.generative_ai.image_generators.sthocastic_image_generator import StochasticImageGenerator
from core.dependencies.ai.generative_ai.image_harmonizers.image_harmonizer import ImageHarmonizer
from core.dependencies.ai.generative_ai.image_harmonizers.mock_image_harnonizer import MockImageHarmonizer
from core.dependencies.ai.generative_ai.image_inpainters.image_inpainter import ImageInpainter
from core.dependencies.ai.generative_ai.image_inpainters.stable_diffusion_image_inpainter import \
    StableDiffusionImageInpainter
from core.dependencies.utils.datasets.bounding_box_generator import BoundingBoxGenerator
from core.dependencies.utils.image.image_editors.image_paster import ImagePaster
from core.dependencies.utils.image.mask_generators.alpha_mask_generator import AlphaMaskGenerator
from core.pipelines.pipeline import DatasetGenerationPipeline, BoundingBox
from core.utils import plot_images


class HarmonizationDatasetGenerator(DatasetGenerationPipeline):
    boat_size : int = 500
    crop_resolution : Tuple[int, int] = (512, 512)
    boat_generator : ImageGenerator
    background_generator : ImageGenerator
    background_remover : BackgroundRemover
    point_extractor : PointExtractor
    harmonizer : ImageHarmonizer
    inpainter: ImageInpainter

    def show(self, img):
        plt.imshow(img)
        plt.show()

    def generate(self) -> Tuple[Image.Image, List[BoundingBox]]:
        bg = self.background_generator.generate()
        points = self.point_extractor.extract(bg)
        bboxs=[]
        aggregate_boats = Image.new("RGBA", bg.size, (0, 0, 0, 0))

        for point in points:
            boat_bg = Image.new("RGBA", bg.size, (0, 0, 0, 0))
            gen_boat = self.boat_generator.generate()
            boat = self.background_remover\
                .remove(self.boat_generator.generate())\
                .resize((self.boat_size, ((gen_boat.size[1]*self.boat_size)//gen_boat.size[0]))).convert("RGBA")
            boat_bg = ImagePaster(center=point, patch=boat).edit(boat_bg)
            aggregate_boats = ImagePaster(center=point, patch=boat).edit(aggregate_boats)
            bbox_generator = BoundingBoxGenerator()
            bboxs.append(bbox_generator.generate(alpha_img=boat_bg))
            bbox_generator.paint(boat_bg, bboxs[-1], color=(255, 0, 0))

        mask = AlphaMaskGenerator(alpha_image=aggregate_boats, type=AlphaMaskGenerator.Type.outside, strength=255).generate()
        boats_with_bg = ImagePaster(center=(bg.size[0]//2, bg.size[1]//2), patch=aggregate_boats).edit(bg)
        boats_inpainted = self.inpainter.inpaint(boats_with_bg, mask)
        plot_images([
            aggregate_boats,
            mask,
            boats_with_bg,
            boats_inpainted,
        ],
        [
            "Aggregate boats",
            "mask",
            "boats with background",
            "boats inpainted"
        ],
        save_as="result_pipeline.png")
        return boats_inpainted, bboxs

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


def generate_dataset(boat_class: str, boat_description: str, number_of_boats: int, environment_description: str, boat_size: int=550):
    img, bboxs = HarmonizationDatasetGenerator(
        boat_size=boat_size,
        crop_resolution=(712, 712),
        boat_generator=StableDiffusionImageGenerator(api_key=os.getenv("SD_API_KEY", ""),
                                                     prompt=boat_description,
                                                     negative_prompt=""),
        background_generator=StableDiffusionImageGenerator(api_key=os.getenv("SD_API_KEY", ""),prompt=environment_description),
        background_remover=StableDiffusionBackgroundRemover(api_key=os.getenv("SD_API_KEY", "")),
        point_extractor=SparsePointExtractor(min_distance_between_points=350, separation_from_edges=200,
                                             initial_point=(800, 800), n_points=number_of_boats, max_distance_between_points=600),
        harmonizer=MockImageHarmonizer(),
        inpainter=StableDiffusionImageInpainter(api_key=os.getenv("SD_API_KEY", ""),
                                                prompt=environment_description,
                                                negative_prompt="boats")
    ).generate()
    [BoundingBoxGenerator().paint(img, bbox, color=(255, 0, 0)) for bbox in bboxs]
    plt.imshow(img)
    plt.show()
    img.save(uuid.uuid4().hex + ".png")


generate_dataset(
                boat_class="fishing ship",
                boat_description="a blue and red coloured big fishing boat viewed from top",
                environment_description="a totally bloody brilliant red ocean viewed from top in a top view perspective",
                number_of_boats=1,
                boat_size=500
             )