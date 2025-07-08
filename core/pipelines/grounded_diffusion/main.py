from time import sleep

from pydantic import BaseModel
from ..pipeline import DatasetGenerationPipeline, PipelineResult, Annotation
from ...dependencies.ai.discriminative_ai.grounded_segmentator.grounded_segmentator import GroundedSegmentator
from ...dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator
from ...utils import polygon_to_bbox
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


class GroundedDiffusionConditioning(BaseModel):
    prompt: str
    negative_prompt: str = ""
    ground_truth: int
    generic_class: str


class GroundedDiffusionPipeline(DatasetGenerationPipeline[GroundedDiffusionConditioning], BaseModel):
    image_generator: ImageGenerator
    grounded_segmentator: GroundedSegmentator

    def generate(self, conditioning: GroundedDiffusionConditioning) -> PipelineResult:
        with Progress(
            TextColumn("[bold]{task.fields[status]}"),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task = progress.add_task(
                "Running Grounded Diffusion Pipeline",
                total=100,
                status="Starting generation"
            )

            progress.update(task, status="💭 Generating image...")
            image = self.image_generator.generate(conditioning.prompt, conditioning.negative_prompt)
            #progress.update(task, advance=30)

            # Step 2: Run segmentation
            progress.update(task, status="🧠 Segmenting image...")
            labels = self.grounded_segmentator.segment(image, conditioning.generic_class)
            progress.update(task, advance=30)

            # Step 3: Convert to bboxes
            progress.update(task, status="📐 Converting polygons to bounding boxes...")
            bboxes = list(map(lambda x: polygon_to_bbox(x.data.points), labels.annotations))
            progress.update(task, advance=20)

            # Step 4: Build annotations
            progress.update(task, status="📦 Building annotations...")
            annotations = [
                Annotation(
                    type="bounding_box",
                    label=conditioning.ground_truth,
                    data=bbox
                ) for bbox in bboxes
            ]
            progress.update(task, advance=20)

        return PipelineResult(
            image=image,
            annotations=annotations
        )
