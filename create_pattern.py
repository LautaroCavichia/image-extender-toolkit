from PIL import Image

def create_pattern(image: Image.Image) -> Image.Image:
    """
    Tiles the input image to fill a 16:9 canvas.
    """
    target_ratio = 9/16
    
    # we make the image small so we can repeat it
    image = image.resize((int(image.width / 10), int(image.height / 10)))
    
    # Get the dimensions of the image
    tile_width, tile_height = image.size
    
    

    
    # We’ll choose a “large enough” 16:9 for demonstration
    # Example: width = 1920, height = 1080
    # Adjust as needed or accept user input for custom size
    base_width = 1920
    base_height = 1080
    
    # Create a new blank image
    pattern_image = Image.new('RGBA', (base_width, base_height), (255, 255, 255, 0))
    
    # Tile the image across the base image
    y = 0
    while y < base_height:
        x = 0
        while x < base_width:
            pattern_image.paste(image, (x, y))
            x += tile_width + 10
        y += tile_height
    
    return pattern_image