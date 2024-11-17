from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class InferenceDataset(Dataset):
    def __init__(self, paths: List[str], size: int = 224):
        self.paths = paths
        self.resize = A.Resize(height=size, width=size)
        self.to_image = v2.ToImage()
        self.to_dtype = v2.ToDtype(torch.float32, scale=True)

    def __getitem__(self, item):
        image = np.array(Image.open(self.paths[item]))
        image = self.resize(image=image)["image"]

        if len(image.shape) >= 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = np.expand_dims(image, axis=2)
        image = self.to_image(image)
        image = self.to_dtype(image)

        return image, self.paths[item]

    def __len__(self):
        return len(self.paths)
