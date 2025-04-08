import base64
import uuid
from datetime import datetime

import cv2

import numpy as np
from PIL import Image
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import os
from io import BytesIO
from typing import Optional, List, Dict

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

from core.dependencies.ai.generative_ai.image_control.stable_diffusion_image_control_editor import \
    StableDiffusionImageControlEditor
from core.dependencies.ai.generative_ai.image_generators.stable_diffusion_image_generator import \
    StableDiffusionImageGenerator


load_dotenv()
app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateImageBody(BaseModel):
    number_of_images: int
    aspect_ratio: str
    base_image : Optional[str] = None
    model_to_use: str
    prompt :str
    negative_prompt: str

@app.post("/generate_images/")
async def generate_image(body: GenerateImageBody):
    image_generator = StableDiffusionImageGenerator(
        api_key=os.getenv("SD_API_KEY", ""),
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        aspect_ratio=body.aspect_ratio
    )
    image = image_generator.generate()
    buffer = BytesIO()
    image.save(buffer, "jpeg")
    return {
        "image": "data:image/jpeg;base64,"+ str(base64.b64encode(buffer.getvalue()))[2:-1],
    }


class Label(BaseModel):
    id: int
    name: str
    prompt: str


class GenerateGroundedImage(BaseModel):
    number_of_images: int
    aspect_ratio: str
    text_to_image_model: str
    annotation_model: str
    prompt: str
    negative_prompt: str
    labels: List[Label]


@app.post("/generate_images/pipeline_grounded_sam/")
async def generate_pipeline_grounded_sam(body: GenerateGroundedImage):
    # Generate image using Stable Diffusion
    image_generator = StableDiffusionImageGenerator(
        api_key=os.getenv("SD_API_KEY", ""),
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        aspect_ratio=body.aspect_ratio
    )
    image = image_generator.generate()  # PIL Image

    # Save image to buffer for output
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    # Convert image to numpy array and get dimensions
    np_image = np.array(image)
    h_img, w_img, _ = np_image.shape

    # Build ontology from provided labels: map prompt -> label id
    ontology_dict = {label.prompt: label.id for label in body.labels}
    # Map GroundedSAM class indices to our label ids (assumes order)
    class_id_to_label = {i: label.id for i, label in enumerate(body.labels)}

    # Run GroundedSAM with our ontology
    gs_model = GroundedSAM(ontology=CaptionOntology(ontology_dict))
    results = gs_model.predict(np_image)

    annotations = []
    # Branch: polygon if "grounded sam", else bounding box
    if body.annotation_model.lower() == "grounded_sam":
        # Create polygon annotations using mask contours
        for i, mask in enumerate(results.mask):
            mask_uint8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            normalized_points = []
            for pt in points:
                x_norm = pt[0] / w_img
                y_norm = pt[1] / h_img
                normalized_points.append([x_norm, y_norm])
            # Use min x and y as left and top
            xs = [pt[0] for pt in normalized_points]
            ys = [pt[1] for pt in normalized_points]
            left = min(xs) if xs else 0.0
            top = min(ys) if ys else 0.0
            label_id = class_id_to_label.get(int(results.class_id[i]), body.labels[0].id)
            annotations.append({
                "id": str(uuid.uuid4()),
                "type": "polygon",
                "is_synthetic": False,
                "data": {
                    "left": left,
                    "top": top,
                    "points": normalized_points
                },
                "label": label_id
            })
    else:
        # Create bounding box annotations using results.xyxy
        for i, box in enumerate(results.xyxy):
            x1, y1, x2, y2 = box
            normalized_x = x1 / w_img
            normalized_y = y1 / h_img
            normalized_width = (x2 - x1) / w_img
            normalized_height = (y2 - y1) / h_img
            label_id = class_id_to_label.get(int(results.class_id[i]), body.labels[0].id)
            annotations.append({
                "type": "bounding_box",
                "id": str(uuid.uuid4()),
                "label": label_id,
                "data": {
                    "point": [normalized_x, normalized_y],
                    "width": normalized_width,
                    "height": normalized_height
                }
            })

    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "image": "data:image/jpeg;base64," + img_str,
        "annotations": annotations
    }

class GenerateImageVariants(BaseModel):
    number_of_images: int
    image: str
    aspect_ratio: str
    prompt: str
    negative_prompt: str
    annotation_model : str
    strength : float = 1
    labels: List[Label]

@app.post("/generate_images/image_variants/")
async def generate_image_variants(body: GenerateImageVariants):
    image_generator = StableDiffusionImageControlEditor(
        api_key=os.getenv("SD_API_KEY", ""),
        control_strength=body.strength
    )
    image_data = base64.b64decode(body.image)
    image = Image.open(BytesIO(image_data))
    image = image_generator.edit(image=image, prompt=body.prompt, negative_prompt=body.negative_prompt)  # PIL Image
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    # Convert image to numpy array and get dimensions
    np_image = np.array(image)
    h_img, w_img, _ = np_image.shape

    # Build ontology from provided labels: map prompt -> label id
    ontology_dict = {label.prompt: label.id for label in body.labels}
    # Map GroundedSAM class indices to our label ids (assumes order)
    class_id_to_label = {i: label.id for i, label in enumerate(body.labels)}

    # Run GroundedSAM with our ontology
    gs_model = GroundedSAM(ontology=CaptionOntology(ontology_dict))
    results = gs_model.predict(np_image)

    annotations = []
    # Branch: polygon if "grounded sam", else bounding box
    if body.annotation_model.lower() == "grounded_sam":
        # Create polygon annotations using mask contours
        for i, mask in enumerate(results.mask):
            mask_uint8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            normalized_points = []
            for pt in points:
                x_norm = pt[0] / w_img
                y_norm = pt[1] / h_img
                normalized_points.append([x_norm, y_norm])
            # Use min x and y as left and top
            xs = [pt[0] for pt in normalized_points]
            ys = [pt[1] for pt in normalized_points]
            left = min(xs) if xs else 0.0
            top = min(ys) if ys else 0.0
            label_id = class_id_to_label.get(int(results.class_id[i]), body.labels[0].id)
            annotations.append({
                "id": str(uuid.uuid4()),
                "type": "polygon",
                "is_synthetic": False,
                "data": {
                    "left": left,
                    "top": top,
                    "points": normalized_points
                },
                "label": label_id
            })
    else:
        for i, box in enumerate(results.xyxy):
            x1, y1, x2, y2 = box
            normalized_x = x1 / w_img
            normalized_y = y1 / h_img
            normalized_width = (x2 - x1) / w_img
            normalized_height = (y2 - y1) / h_img
            label_id = class_id_to_label.get(int(results.class_id[i]), body.labels[0].id)
            annotations.append({
                "type": "bounding_box",
                "id": str(uuid.uuid4()),
                "label": label_id,
                "data": {
                    "point": [normalized_x, normalized_y],
                    "width": normalized_width,
                    "height": normalized_height
                }
            })

    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "image": "data:image/jpeg;base64," + img_str,
        "annotations": annotations
    }
