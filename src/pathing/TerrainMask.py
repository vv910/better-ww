"""Mask high-contrast map HUD markers before extracting terrain features."""

import cv2
import numpy as np


def terrain_feature_mask(image):
    """Return 255 for usable terrain and 0 for icon pixels plus a local safety margin.

    Map screenshots are BGR. Grayscale fixtures still suppress bright white labels and
    stars, while retaining ordinary midtone synthetic terrain.
    """
    if image.ndim == 2:
        candidates = ((image >= 205, 20, 18),)
    elif image.ndim == 3 and image.shape[2] >= 3:
        bgr = image[:, :, :3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        white = (value >= 205) & (saturation <= 75)
        blue = (hue >= 82) & (hue <= 135) & (saturation >= 85) & (value >= 65)
        gold = (hue >= 8) & (hue <= 40) & (saturation >= 90) & (value >= 90)
        color_padding = max(8, min(18, round(min(image.shape[:2]) * .012)))
        candidates = ((white, 20, 18), (blue, 80, color_padding), (gold, 80, color_padding))
    else:
        raise ValueError('Terrain feature mask expects a grayscale or BGR image')

    # Large connected color regions are often actual map terrain (water, snow,
    # or broad shading); tiny saturated flecks are terrain texture. Keep only
    # compact, marker-sized connected regions.
    area_limit = min(10000, max(900, round(image.shape[0] * image.shape[1] * .01)))
    selected = np.zeros(image.shape[:2], dtype=np.uint8)
    for candidate, minimum_area, padding in candidates:
        binary = candidate.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        keep = np.zeros(count, dtype=np.uint8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep[1:] = ((areas >= minimum_area) & (areas <= area_limit)).astype(np.uint8)
        excluded = keep[labels] * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
        selected |= cv2.dilate(excluded, kernel)
    return cv2.bitwise_not(selected)
