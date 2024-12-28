import os
import random
import urllib.request
import io

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionInpaintPipeline

def setup_pipeline():
    """
    Sets up the Stable Diffusion Inpaint Pipeline.

    Returns:
        StableDiffusionInpaintPipeline: The initialized inpaint pipeline.
    """
    model_name = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
    return pipe

def extend_ai(
    input_image: Image.Image,
    prompt: str = "",
    negative_prompt: str = "",
    num_inference_steps: int = 4,
    guidance_scale: float = 7.5,
    transition_size: int = 256,
    direction: str = "top",
    expand_pixels: int = 256
) -> Image.Image:
    """
    Extend the image in the specified direction using inpainting.

    Args:
        pipe (StableDiffusionInpaintPipeline): The inpainting pipeline.
        input_image (PIL.Image.Image): The original image to be extended.
        prompt (str): Prompt to guide the inpainting.
        negative_prompt (str): Negative prompt to reduce unwanted artifacts.
        num_inference_steps (int): Number of diffusion steps for inpainting.
        guidance_scale (float): Prompt guidance scale.
        transition_size (int): Size of the smooth blending zone.
        direction (str): Direction to extend ('top', 'bottom', 'left', 'right').
        expand_pixels (int): Number of pixels to expand in each iteration.

    Returns:
        PIL.Image.Image: The extended image.
    """
    
    pipe = setup_pipeline()
    
    width, height = input_image.size
  
    # Determine new dimensions based on direction
    if direction in ["top", "bottom"]:
        new_width = width
        new_height = height + expand_pixels
    elif direction in ["left", "right"]:
        new_width = width + expand_pixels
        new_height = height
    else:
        raise ValueError("Invalid direction. Choose from 'top', 'bottom', 'left', 'right'.")

    # Create a new blank image with the new dimensions
    extended_image = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

    # Paste the original image onto the new canvas
    if direction == "top":
        extended_image.paste(input_image, (0, expand_pixels))
    elif direction == "bottom":
        extended_image.paste(input_image, (0, 0))
    elif direction == "left":
        extended_image.paste(input_image, (expand_pixels, 0))
    elif direction == "right":
        extended_image.paste(input_image, (0, 0))

    # Create a mask for the region to be inpainted
    mask = Image.new("L", (new_width, new_height), color=0)
    mask_pixels = np.array(mask)

    if direction == "top":
        mask_pixels[0:expand_pixels, :] = 255
    elif direction == "bottom":
        mask_pixels[-expand_pixels:, :] = 255
    elif direction == "left":
        mask_pixels[:, 0:expand_pixels] = 255
    elif direction == "right":
        mask_pixels[:, -expand_pixels:] = 255

    # Add a smooth transition at the boundary
    if transition_size > 0:
        if direction in ["top", "bottom"]:
            for y in range(expand_pixels - transition_size, expand_pixels):
                blend_ratio = 1 - ((y - (expand_pixels - transition_size)) / transition_size)
                blend_ratio = np.clip(blend_ratio, 0, 1)
                if direction == "top":
                    mask_pixels[expand_pixels + y, :] = int(255 * blend_ratio)
                else:  # bottom
                    mask_pixels[new_height - expand_pixels + y, :] = int(255 * blend_ratio)
        elif direction in ["left", "right"]:
            for x in range(expand_pixels - transition_size, expand_pixels):
                blend_ratio = (x - (expand_pixels - transition_size)) / transition_size
                blend_ratio = np.clip(blend_ratio, 0, 1)
                if direction == "left":
                    mask_pixels[:, x] = int(255 * blend_ratio)
                else:  # right
                    mask_pixels[:, new_width - expand_pixels + x] = int(255 * blend_ratio)

    mask = Image.fromarray(mask_pixels, mode="L")
    
  
    
    #round to nearest multiple of 8 
    
    new_width = (new_width + 7) & ~7
    new_height = (new_height + 7) & ~7

    print("prompt", prompt)
    mask.save(os.path.join("mask.png"))
    extended_image.save(os.path.join("extended_image.png"))

    # Run inpainting
    inpainted_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=extended_image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=new_width,
        height=new_height,
    ).images[0]

    # To maintain original image dimensions, crop or adjust as needed
    if direction in ["top", "bottom"]:
        # For vertical expansion
        if direction == "top":
            # Crop to keep the extended area and original image
            combined_image = Image.new("RGB", (new_width, new_height))
            combined_image.paste(inpainted_image, (0, 0))
        else:  # bottom
            combined_image = Image.new("RGB", (new_width, new_height))
            combined_image.paste(inpainted_image, (0, 0))
    elif direction in ["left", "right"]:
        # For horizontal expansion
        if direction == "left":
            combined_image = Image.new("RGB", (new_width, new_height))
            combined_image.paste(inpainted_image, (0, 0))
        else:  # right
            combined_image = Image.new("RGB", (new_width, new_height))
            combined_image.paste(inpainted_image, (0, 0))
    
    return inpainted_image

def iterative_extend(
    pipe: StableDiffusionInpaintPipeline,
    input_image: Image.Image,
    prompt: str = "sky",
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    transition_size: int = 32,
    direction: str = "top",
    expand_pixels: int = 256,
    times_to_expand: int = 4
) -> Image.Image:
    """
    Iteratively extend the image multiple times in the specified direction.

    Args:
        pipe (StableDiffusionInpaintPipeline): The inpainting pipeline.
        input_image (PIL.Image.Image): The original image to be extended.
        prompt (str): Prompt to guide the inpainting.
        negative_prompt (str): Negative prompt to reduce unwanted artifacts.
        num_inference_steps (int): Number of diffusion steps for inpainting.
        guidance_scale (float): Prompt guidance scale.
        transition_size (int): Size of the smooth blending zone.
        direction (str): Direction to extend ('top', 'bottom', 'left', 'right').
        expand_pixels (int): Number of pixels to expand in each iteration.
        times_to_expand (int): Number of times to perform the extension.

    Returns:
        PIL.Image.Image: The fully extended image after all iterations.
    """
    final_image = input_image
    for i in range(times_to_expand):
        print(f"Extension {i+1}/{times_to_expand}: Extending {direction}")
        final_image = extend_ai(
            pipe=pipe,
            input_image=final_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            transition_size=transition_size,
            direction=direction,
            expand_pixels=expand_pixels
        )
    return final_image

def download_image(url: str) -> Image.Image:
    """
    Downloads an image from a URL and returns it as a PIL Image.

    Args:
        url (str): URL of the image to download.

    Returns:
        PIL.Image.Image: The downloaded image.
    """
    with urllib.request.urlopen(url) as url_response:
        img_data = url_response.read()
    input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
    return input_image