import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from collections import Counter
from PIL import Image
import io
from rembg import remove
import base64
from pyinpaint import Inpaint
import os

app = Flask(__name__)

TARGET_WIDTH = 1800
TARGET_HEIGHT = 3200


def expand_with_edge_blur(image, blur_strength, deadzone_percent):
    """
    Resize the image to the target size with aggressive radial edge blur or plain background handling.
    If plain_background_mode is True, the background will be filled with a solid color or gradient.
    """
    # Convert PIL Image to OpenCV format
    image = np.array(image.convert("RGBA"))
    original_height, original_width = image.shape[:2]

    # Early return for already-large images
    if original_width >= TARGET_WIDTH and original_height >= TARGET_HEIGHT:
        return Image.fromarray(image)

    offset_ratio = 0.15 
    total_vertical_padding = TARGET_HEIGHT - original_height
    top_pad = int(total_vertical_padding * (0.5 + offset_ratio))
    bottom_pad = total_vertical_padding - top_pad

    total_horizontal_padding = TARGET_WIDTH - original_width
    left_pad = total_horizontal_padding // 2
    right_pad = total_horizontal_padding - left_pad

   
        # Use reflection (default mode)
    expanded_image = cv2.copyMakeBorder(
        image, top_pad, bottom_pad, left_pad, right_pad, 
        borderType=cv2.BORDER_REFLECT
        )  

    height, width = expanded_image.shape[:2]
    center_y, center_x = height // 2, width // 2

    # Create optimized distance map using vectorized operations
    Y, X = np.ogrid[:height, :width]
    dist_map = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_distance = np.sqrt(center_y**2 + center_x**2)

    # Calculate deadzone
    deadzone_radius = int(min(original_width, original_height) * float(deadzone_percent) / 100)
    
    # Create a more aggressive blur mask
    normalized_dist = (dist_map - deadzone_radius) / (max_distance - deadzone_radius)
    mask = np.clip(normalized_dist, 0, 1) ** 2

    # Ensure valid odd kernel sizes
    def get_valid_kernel_size(size):
        size = max(1, size)  # Ensure size is positive
        return size if size % 2 == 1 else size + 1  # Make sure it's odd
    
    # Create multiple blur levels for progressive blurring
    blur_levels = [
        cv2.GaussianBlur(expanded_image, (get_valid_kernel_size(int(blur_strength) // 4),
                                          get_valid_kernel_size(int(blur_strength) // 4)), 0),
        cv2.GaussianBlur(expanded_image, (get_valid_kernel_size(int(blur_strength) // 2),
                                          get_valid_kernel_size(int(blur_strength) // 2)), 0),
        cv2.GaussianBlur(expanded_image, (get_valid_kernel_size(int(blur_strength)),
                                          get_valid_kernel_size(int(blur_strength))), 0)
    ]

    # Blend multiple blur levels based on distance
    result = expanded_image.copy()
    for i, blurred in enumerate(blur_levels):
        blend_mask = np.clip((mask - (i / len(blur_levels))) * len(blur_levels), 0, 1)
        result = (blurred * blend_mask[..., None] + 
                 result * (1 - blend_mask[..., None])).astype(np.uint8)

    return Image.fromarray(result)

def get_dominant_color(image):
    # Convert image to RGB
    image_rgb = image.convert('RGB')
    
    # Get the image data as a flattened list of RGB tuples
    pixels = np.array(image_rgb).reshape(-1, 3)
    
    # Find the most common color in the image (using Counter to get frequency of colors)
    colors = Counter(map(tuple, pixels))
    dominant_color = colors.most_common(1)[0][0]
    
    return dominant_color

# Function to expand the image with the dominant background color
def expand_with_dominant_color(image, target_width, target_height):
    # Get the dominant background color
    dominant_color = get_dominant_color(image)
    
    # Create a new image with the target dimensions and the dominant background color
    expanded_image = Image.new('RGB', (target_width, target_height), dominant_color)
    
    # Paste the original image in the center of the new image
    # Calculate the offset for pasting
    offset_ratio = 0.15  # 20% offset for more space at the top
    vertical_padding = target_height - image.height
    horizontal_padding = target_width - image.width

    # Adjust the Y-coordinate for the offset
    y_offset = int(vertical_padding * (0.5 + offset_ratio))  # Shift down by 20% more space at the top
    x_offset = horizontal_padding // 2  # Centered horizontally

    # Paste the original image at the adjusted position
    expanded_image.paste(image, (x_offset, y_offset))
    
    return expanded_image

def upscale_image(image, min_height):
    # Get the current width and height of the image
    width, height = image.size
    
    # If the height is less than the minimum required height
    if height < min_height:
        # Calculate the new width to maintain the aspect ratio
        new_height = min_height
        new_width = int((new_height / height) * width)
        
        # Resize the image while maintaining the aspect ratio
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return image


def remove_background_precise(image, mask=None):
    """
    Removes the background using rembg and optionally refines with a mask.
    
    Args:
        image: PIL Image object.
        mask: Optional base64 string of the user-painted mask.
    
    Returns:
        PIL Image with background refined using the mask.
    """
    try:
        # Convert the input image to a numpy array
        input_array = np.array(image.convert("RGBA"))
        
        # Step 1: Perform initial background removal with rembg
        bg_removed_array = remove(input_array)
        
        # Ensure the array is writable (make a copy)
        bg_removed_array = bg_removed_array.copy()

        # Step 2: Apply additional mask refinement if a mask is provided
        if mask:
            # Decode and process the mask
            mask_array = process_mask_data(mask)

            # Ensure mask dimensions match the image
            mask_array = cv2.resize(mask_array, (bg_removed_array.shape[1], bg_removed_array.shape[0]))

            # Apply the mask to refine the removal
            # (Set the masked areas to transparent)
            bg_removed_array[mask_array == 255] = [0, 0, 0, 0]  # Transparent

        # Convert the result back to a PIL Image
        refined_image = Image.fromarray(bg_removed_array)

        return refined_image

    except Exception as e:
        print(f"Error during background removal: {e}")
        return image  # Return original image if an error occurs


def process_mask_data(mask_data_url):
    """
    Convert base64 mask data to binary numpy array compatible with OpenCV.

    Args:
        mask_data_url: Base64-encoded mask image URL
    Returns:
        Binary mask as a numpy array (uint8)
    """
    try:
        # Remove the data URL prefix
        mask_base64 = mask_data_url.split(',')[1]
        # Decode base64 to bytes
        mask_bytes = base64.b64decode(mask_base64)
        # Convert to numpy array
        mask_arr = np.frombuffer(mask_bytes, np.uint8)
        # Decode image
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_UNCHANGED)

        # Convert to grayscale if necessary
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)

        # Ensure mask is binary (0 for inpaint region, 255 for background)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        return mask.astype(np.uint8)
    except Exception as e:
        print(f"Error processing mask: {str(e)}")
        raise
def remove_object(image, mask_data):
    """
    Remove object from image using OpenCV's inpainting methods.

    Args:
        image: PIL Image object.
        mask_data: base64 string of mask data.
    Returns:
        PIL Image with object removed.
    """
    try:
        # Convert PIL Image to OpenCV format (BGR)
        org_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Process the mask
        mask = process_mask_data(mask_data)

        # Ensure mask dimensions match the original image
        mask = cv2.resize(mask, (org_img.shape[1], org_img.shape[0]))

        # Perform inpainting using OpenCV
        inpainted_img = cv2.inpaint(org_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

        # Convert the result back to PIL Image
        result = Image.fromarray(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))

        return result

    except Exception as e:
        print(f"Error in remove_object: {str(e)}")
        raise

    except Exception as e:
        print(f"Error in remove_object: {str(e)}")
        raise
    
@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get the uploaded image
    file = request.files['image']
    image = Image.open(file).convert("RGB")  # Convert to RGB for consistency

    # Get the operation from the form data
    operation = request.form.get('operation', 'resize')
    
    image = upscale_image(image, 1000)

    try:
        # Get mask data from form if available
        mask_data = request.form.get('mask')

        if operation == 'remove_object':
            if not mask_data:
                return jsonify({"error": "No mask data provided for remove_object operation"}), 400
            processed_image = remove_object(image, mask_data)

        elif operation == 'remove_bg_precise':
            # Process image with or without mask for precise background removal
            processed_image = remove_background_precise(image, mask_data)

        elif operation == 'expand_image':
            # Expand image with edge blur
            blur_strength = int(request.form.get('blur_strength', 300))
            deadzone = int(request.form.get('deadzone', 20))
            processed_image = expand_with_edge_blur(image, blur_strength, deadzone)

        elif operation == 'expand_color':
            # Expand image using dominant color
            target_width = int(request.form.get('target_width', 800))
            target_height = int(request.form.get('target_height', 600))
            processed_image = expand_with_dominant_color(image, target_width, target_height)

        elif operation == 'resize':
            # Resize image logic
            target_width = int(request.form.get('target_width', 800))
            target_height = int(request.form.get('target_height', 600))
            processed_image = image.resize((target_width, target_height))

        else:
            return jsonify({"error": "Invalid operation"}), 400

        # Save the processed image to a byte stream and send it back
        img_io = io.BytesIO()
        processed_image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png', as_attachment=False)

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500



@app.route('/')
def index():
    return send_file('templates/index.html')


if __name__ == '__main__':
    app.run(debug=True)