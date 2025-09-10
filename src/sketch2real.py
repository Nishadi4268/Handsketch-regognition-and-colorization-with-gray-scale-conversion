# src/sketch2real.py

import cv2
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

def sketch_to_real(sketch_path, output_path, prompt_text, device="cuda"):
    """
    Convert a sketch image to a realistic RGB image using ControlNet.

    Args:
        sketch_path (str): Path to the input sketch image.
        output_path (str): Path to save the generated realistic image.
        prompt_text (str): Text prompt describing the image (can use predicted label).
        device (str): 'cuda' or 'cpu'.
    
    Returns:
        str: Path to the generated realistic image.
    """
    # Load ControlNet model
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.to(device)

    # Load sketch
    sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)  # convert to 3 channels
    sketch = Image.fromarray(sketch)

    # Generate realistic image
    output = pipe(prompt=prompt_text, image=sketch, num_inference_steps=20).images[0]

    # Save output
    output.save(output_path)
    return output_path
