from PIL import Image
import os
from torch.utils.data import Dataset

class Unlabeled_dataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.image_paths = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(("png", "jpg", "jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image