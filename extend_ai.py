import PIL
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

model_name: str = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")

import numpy as np

def extend_ai(
    input_image: Image.Image,
    prompt: str = "sky",
    target_ratio: float = 1.7,
    
    negative_prompt: str = "",
    num_inference_steps: int = 9,
    guidance_scale: float = 7.5,
    transition_size: int = 32,
):
    """
    Extend the height of an image by 'target_ratio', adding generated content to the top
    while keeping the original content untouched and at the bottom.
    
    Args:
        input_image (PIL.Image): The original image to be extended.
        pipe (StableDiffusionInpaintPipeline): Stable Diffusion pipeline for inpainting.
        target_ratio (float): Target height ratio (new_height = original_height * target_ratio).
        prompt (str): Prompt to guide the inpainting.
        negative_prompt (str): Negative prompt to reduce unwanted artifacts.
        num_inference_steps (int): Number of diffusion steps for inpainting.
        guidance_scale (float): Prompt guidance scale.
        transition_size (int): Height of the smooth blending zone.
    
    Returns:
        PIL.Image: The image with new content added at the top.
    """
    # Original dimensions
    width, height = input_image.size
    new_height = int(height * target_ratio)

    if new_height <= height:
        # No extension needed
        return input_image

    # Create a blank canvas at the new size
    extended_image = Image.new("RGB", (width, new_height), color=(0, 0, 0))
    # Paste the original image at the bottom
    extended_image.paste(input_image, (0, new_height - height))

    # Create a mask: the top region (added part) is white (255), rest is black (0)
    mask = Image.new("L", (width, new_height), color=0)
    mask_pixels = np.array(mask)
    mask_pixels[: new_height - height, :] = 255  # Top region for inpainting

    # Add a smooth transition at the boundary
    for y in range(new_height - height, min(new_height - height + transition_size, new_height)):
        blend_ratio = 1 - (y - (new_height - height)) / transition_size
        mask_pixels[y, :] = int(255 * blend_ratio)

    mask = Image.fromarray(mask_pixels, mode="L")

    # Run inpainting
    inpainted_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=extended_image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # Return the final inpainted image
    return inpainted_image

