import cv2
import numpy as np
from PIL import Image

def extend(
    pil_image: Image.Image,
    top_extension_ratio: float = 0.75,
    mirror_band_ratio: float = 0.1,
    inpaint_radius: float = 4.0,
    inpaint_method: int = cv2.INPAINT_NS
) -> Image.Image:
    """
    Extends the image height to (1 + top_extension_ratio) * original_height.
    For top_extension_ratio=0.75, total height = 1.75 × original.
    
    Steps:
      1) Original at bottom.
      2) Mirrored band (vertical flip) from the top portion of the original
         placed right above the original top edge.
      3) Inpaint the remaining top area.
      4) Enhanced blending:
         - partial fade in the mask boundary for smoother inpainting
         - alpha blend the inpainted region with the mirrored band.
    
    :param pil_image:         Input PIL image
    :param top_extension_ratio:
        e.g. 0.75 => final height = 1.75 × original
    :param mirror_band_ratio:
        Fraction of the original height to vertically flip. e.g. 0.2 => 20%.
    :param inpaint_radius:    Neighborhood radius for cv2.inpaint
    :param inpaint_method:    cv2.INPAINT_TELEA (default) or cv2.INPAINT_NS
    :return:                  Extended PIL image
    """

    # Convert PIL → NumPy (RGB)
    img = np.array(pil_image.convert("RGB"))
    orig_h, orig_w, _ = img.shape

    # Calculate new total height
    added_h = int(round(orig_h * top_extension_ratio))  # extra space on top
    new_h = orig_h + added_h  # e.g. 1.75 × orig_h if ratio=0.75
    if added_h <= 0:
        raise ValueError("top_extension_ratio must be > 0.")

    # Prepare extended canvas (black)
    extended = np.zeros((new_h, orig_w, 3), dtype=np.uint8)

    # 1) Place original at the bottom: from row=added_h to row=added_h + orig_h
    extended[added_h : added_h + orig_h, :, :] = img

    # 2) Create the mirrored band (vertical flip) from the top patch
    mirror_h = int(round(orig_h * mirror_band_ratio))
    if mirror_h <= 0 or mirror_h >= orig_h:
        raise ValueError("mirror_band_ratio must be between 0 and 1 (e.g., 0.2).")

    top_patch = img[0 : mirror_h, :, :]          # top portion of original
    top_patch_flipped = cv2.flip(top_patch, 0)   # flip vertically (flipCode=0)

    # Place it so the band’s bottom edge aligns with the original top
    mirror_start = added_h - mirror_h
    mirror_end = added_h
    if mirror_start < 0:
        raise ValueError("Mirror band doesn't fit. Increase top_extension_ratio or reduce mirror_band_ratio.")

    extended[mirror_start : mirror_end, :, :] = top_patch_flipped

    # -------------------------
    # (A) PARTIAL MASK FADE for inpainting
    # We'll inpaint everything above the mirrored band => [0 : mirror_start].
    # But we also fade the boundary so that it's not a hard line in the mask.
    # -------------------------
    mask = np.zeros((new_h, orig_w), dtype=np.uint8)
    # Everything above the mirror band is missing
    mask[0 : mirror_start, :] = 255

    # Fade the last 10px near mirror_start from 255 -> 0
    fade_px = 10
    fade_start = max(mirror_start - fade_px, 0)
    for row in range(fade_start, mirror_start):
        alpha = (row - fade_start) / float(mirror_start - fade_start)  # 0→1
        val = int(255 * (1 - alpha))  # 255 => 0
        mask[row, :] = val

    # 3) Inpaint
    extended_bgr = cv2.cvtColor(extended, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(extended_bgr, mask, inpaint_radius, inpaint_method)
    inpainted = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    # -------------------------
    # (B) FINAL ALPHA BLEND:
    #    In the region [mirror_start - blend_h : mirror_start],
    #    we merge the mirrored band in 'extended' with the inpaint result
    #    for a smoother seam.
    # -------------------------
    blend_h = 0  # how many rows to blend
    blend_start = max(mirror_start - blend_h, 0)
    blend_end = mirror_start

    # We'll combine:
    #   mirrored band from 'extended'   (call it base_region)
    #   inpainted result from 'inpainted' (call it new_region)
    # so that row=blend_start => 100% base, row=blend_end => 100% inpainted
    for row in range(blend_start, blend_end):
        alpha = (row - blend_start) / float(blend_end - blend_start)  # 0→1
        base_row = extended[row, :, :].astype(np.float32)   # mirrored band
        new_row  = inpainted[row, :, :].astype(np.float32)  # inpaint
        blended_row = (1 - alpha) * base_row + alpha * new_row
        inpainted[row, :, :] = blended_row.astype(np.uint8)

    # Convert final result to PIL
    result = Image.fromarray(inpainted, mode="RGB")
    return result
