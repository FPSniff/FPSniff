import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FingerprintDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith('.bmp'):
                    self.img_paths.append(os.path.join(root, file))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("L")  
        if self.transform:
            img = self.transform(img)
        folder_name = os.path.basename(os.path.dirname(img_path))
        folder_name_numeric = int(folder_name)
        label = torch.tensor(folder_name_numeric)
        return img, label