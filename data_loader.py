import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Tuple

### 1. BUILD A CUSTOM DATASET
class ImageDataset(Dataset):
    """
    We'll make a custom dataset that'll;
    1. Take images, labels, and transform as the arguments.
    2. Overwrite the` __getitem__()` and `__len__()` methods
    3. Apply the transformation if given.
    4. Return the data and labels as tuple
    """

    # Define a constructor
    def __init__(self, data_path: List, labels, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load the image
        img = Image.open(self.data_path[idx])
        
        # Get the respective label 
        label = self.labels[idx]

        # Apply the defined transformation
        if self.transform:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.data_path)

