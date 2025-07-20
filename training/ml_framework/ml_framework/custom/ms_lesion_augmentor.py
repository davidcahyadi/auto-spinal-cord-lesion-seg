import random

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class MSLesionAugmentor:
    def __init__(
        self,
        min_lesions=1,
        max_lesions=3,
        min_size=5,
        max_size=20,
    ):
        self.min_lesions = min_lesions
        self.max_lesions = max_lesions
        self.min_size = min_size
        self.max_size = max_size

    def _create_lesion(self, size):
        """Create a simple elliptical lesion"""
        mask = np.zeros((size * 3, size * 3), dtype=np.uint8)
        center = (size * 1.5, size * 1.5)
        axes = (size, int(size * random.uniform(0.6, 1.0)))
        angle = random.uniform(0, 360)
        cv2.ellipse(mask, (int(center[0]), int(center[1])), axes, angle, 0, 360, 1, -1)
        return mask

    def _create_texture(self, size):
        """Create a soft, subtle texture for the lesion"""
        texture = np.zeros((size * 3, size * 3), dtype=np.float32)

        # Add subtle random noise for texture
        noise = np.random.normal(0.9, 0.1, texture.shape)  # Reduced variance
        # Apply stronger smoothing for softer texture
        texture = gaussian_filter(noise, sigma=2.0)  # Increased sigma

        # Normalize texture to a narrower range [0.9, 1.1]
        texture = (texture - texture.min()) / (
            texture.max() - texture.min()
        ) * 0.5 + 0.75
        return texture

    def augment(self, image, spinal_cord_mask):
        """Add textured bright lesions strictly within the spinal cord mask"""
        image = image.squeeze()
        spinal_cord_mask = spinal_cord_mask.squeeze()
        H, W = image.shape[:2]
        lesion_mask = np.zeros_like(spinal_cord_mask)
        augmented_image = image.copy()

        # Get reference intensity
        cord_pixels = image[spinal_cord_mask > 0]
        max_intensity = np.median(cord_pixels)
        base_lesion_intensity = min(200, max_intensity * np.random.uniform(1.25, 2.25))
        num_lesions = random.randint(self.min_lesions, self.max_lesions)

        for _ in range(num_lesions):
            lesion_created = False
            try_count = 0
            while not lesion_created and try_count < 10:
                try_count += 1
                size = random.randint(self.min_size, self.max_size)
                lesion = self._create_lesion(size)
                texture = self._create_texture(size)

                # Find valid location within spinal cord
                valid_locations = np.where(spinal_cord_mask > 0)
                if len(valid_locations[0]) == 0:
                    continue

                idx = random.randint(0, len(valid_locations[0]) - 1)
                center_y, center_x = valid_locations[0][idx], valid_locations[1][idx]

                # Place lesion
                y1 = max(0, center_y - size)
                y2 = min(H, center_y + size)
                x1 = max(0, center_x - size)
                x2 = min(W, center_x + size)

                try:
                    lesion_region = lesion[
                        int(y1 - (center_y - size)) : int(y2 - (center_y - size)),
                        int(x1 - (center_x - size)) : int(x2 - (center_x - size)),
                    ]

                    texture_region = texture[
                        int(y1 - (center_y - size)) : int(y2 - (center_y - size)),
                        int(x1 - (center_x - size)) : int(x2 - (center_x - size)),
                    ]

                    # Mask the lesion to stay within spinal cord
                    cord_region = spinal_cord_mask[y1:y2, x1:x2]
                    lesion_region = lesion_region & (cord_region > 0)

                    if np.any(lesion_region):
                        # Create smooth bright lesion with subtle texture
                        smooth_mask = gaussian_filter(
                            lesion_region.astype(float), sigma=1.0
                        )

                        # Apply the bright lesion with subtle texture variation
                        region = augmented_image[y1:y2, x1:x2]
                        if len(region.shape) == 3:
                            smooth_mask = smooth_mask[..., None]
                            texture_region = texture_region[..., None]

                        # Create textured bright lesion with subtle intensity variation
                        lesion_intensity = (
                            base_lesion_intensity * texture_region
                        )  # Texture now varies intensity by ±10%

                        region = smooth_mask * lesion_intensity + (1 - smooth_mask) * region

                        augmented_image[y1:y2, x1:x2] = region
                        lesion_mask[y1:y2, x1:x2] = lesion_region

                except ValueError:
                    continue
                lesion_created = True


        return np.expand_dims(augmented_image,2), np.expand_dims(lesion_mask,2)
