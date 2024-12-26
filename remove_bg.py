# remove_bg.py

import io
from rembg import remove
from PIL import Image

def remove_background(image: Image.Image) -> Image.Image:
    """
    Removes the background from an image using the rembg library, 
    returning a new image with transparent background.
    """
    # Convert PIL Image to bytes in PNG format
    input_bytes = io.BytesIO()
    image.save(input_bytes, format="PNG")
    input_bytes.seek(0)

    # Use rembg to remove background
    output_bytes = remove(input_bytes.getvalue())

    # Convert the result back into a PIL Image
    output_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    return output_image