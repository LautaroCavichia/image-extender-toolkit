import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from collections import Counter
from PIL import Image
import io
from rembg import remove

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


def remove_background_lighter(image):
    # Convert image to numpy array
    image_np = np.array(image)

    # Enhance contrast with CLAHE
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Apply Gaussian Blur to smooth out the image and reduce noise
    image_np = cv2.GaussianBlur(image_np, (5, 5), 0)

    # Generate initial mask with edge detection
    edges = cv2.Canny(image_np, 100, 200)
    mask = np.zeros(image_np.shape[:2], np.uint8)
    mask[edges > 0] = cv2.GC_PR_FGD

    # Rectangle initialization (adjust the rectangle as needed)
    height, width = image_np.shape[:2]
    rect = (20, 20, width - 40, height - 40)

    # Background and foreground models for grabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply grabCut with more iterations for refinement
    cv2.grabCut(image_np, mask, rect, bgd_model, fgd_model, 15, cv2.GC_INIT_WITH_RECT)

    # Refine mask by setting definite foreground and background regions
    mask2 = np.copy(mask)
    mask2[(mask == 2) | (mask == 0)] = 0
    mask2[(mask == 1) | (mask == 3)] = 1

    # Create the final image (foreground with alpha channel)
    result_rgba = np.dstack([image_np, mask2.astype(np.uint8) * 255])

    return Image.fromarray(result_rgba)


def remove_background_precise(image):
    """
    Removes the background using rembg.
    """
    # Convert the input image to a numpy array
    input_array = np.array(image.convert("RGBA"))
    
    # Use rembg to remove the background
    output_array = remove(input_array)
    
    # Convert the output array back to a PIL Image
    output_image = Image.fromarray(output_array)
    
    return output_image


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image = Image.open(file).convert("RGBA")

    # Get the operation from the form data
    operation = request.form.get('operation', 'resize')

    # Preprocess
    image = upscale_image(image, min_height=1000)

    if operation == 'remove_bg_fast':
        processed_image = remove_background_lighter(image)
    elif operation == 'remove_bg_precise':
        processed_image = remove_background_precise(image)
    elif operation == 'expand_image':
        processed_image = expand_with_edge_blur(image, blur_strength=400, deadzone_percent=20)  
    elif operation == 'expand_color':
        processed_image = expand_with_dominant_color(image, TARGET_WIDTH, TARGET_HEIGHT)
    elif operation == 'resize':
        # Default resize operation
        blur_strength = int(request.form.get('blur_strength', 300))
        blur_strength = blur_strength + 1 if blur_strength % 2 == 0 else blur_strength
        deadzone = int(request.form.get('deadzone', 20))
        plain_background_mode = request.form.get('plain_background_mode', 'false') == 'true'
        if plain_background_mode:
            processed_image = expand_with_dominant_color(image, TARGET_WIDTH, TARGET_HEIGHT)
        else:
            processed_image = expand_with_edge_blur(image, blur_strength, deadzone)
    else:
        return jsonify({"error": "Invalid operation"}), 400

    # Save the processed image to a byte stream and send it back
    img_io = io.BytesIO()
    processed_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png', as_attachment=False)


@app.route('/')
def index():
    return send_file('templates/index.html')


if __name__ == '__main__':
    app.run(debug=True)