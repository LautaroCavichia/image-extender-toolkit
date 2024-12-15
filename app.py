import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io

app = Flask(__name__)

TARGET_WIDTH = 1800
TARGET_HEIGHT = 3200

import cv2
import numpy as np
from PIL import Image

def resize_with_edge_blur(image, blur_strength, deadzone_percent, plain_background_mode=False):
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

    # Calculate padding
    top_pad = (TARGET_HEIGHT - original_height) // 2
    bottom_pad = TARGET_HEIGHT - original_height - top_pad
    left_pad = (TARGET_WIDTH - original_width) // 2
    right_pad = TARGET_WIDTH - original_width - left_pad

    if plain_background_mode:
        # Check if the background is plain by inspecting corner pixel values
        top_left = image[0, 0]
        top_right = image[0, original_width - 1]
        bottom_left = image[original_height - 1, 0]
        bottom_right = image[original_height - 1, original_width - 1]
        
        # If all corner pixels are similar, assume plain background
        if np.allclose(top_left, top_right) and np.allclose(top_left, bottom_left) and np.allclose(top_left, bottom_right):
            # Use the background color or gradient for padding
            background_color = top_left  # Assuming top-left is the background color
            expanded_image = np.full((TARGET_HEIGHT, TARGET_WIDTH, 4), background_color, dtype=np.uint8)
        else:
            # Fall back to reflection if not plain background
            expanded_image = cv2.copyMakeBorder(
                image, top_pad, bottom_pad, left_pad, right_pad, 
                borderType=cv2.BORDER_REFLECT
            )
    else:
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



@app.route('/process', methods=['POST'])
@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image = Image.open(file).convert("RGBA")

    # Get the blur strength from the form data, ensuring it is odd
    blur_strength = int(request.form.get('blur_strength', 300))
    blur_strength = blur_strength + 1 if blur_strength % 2 == 0 else blur_strength
    
    # Get the deadzone percentage from the form data
    deadzone = request.form.get('deadzone', 20)
    
    # Check if plain background mode is enabled (from the checkbox in the form)
    plain_background_mode = request.form.get('plain_background_mode', 'false') == 'true'

    # Process the image with the new option for plain background mode
    processed_image = resize_with_edge_blur(image, blur_strength, deadzone, plain_background_mode)

    # Save the processed image to a byte stream and send it back
    img_io = io.BytesIO()
    processed_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


@app.route('/')
def index():
    return send_file('templates/index.html')

if __name__ == '__main__':
    app.run(debug=True)