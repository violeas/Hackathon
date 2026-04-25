import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ChangeDetectionDataset(Dataset):
    def __init__(self, t1_dir, t2_dir, label_dir):
        self.t1_images = sorted(os.listdir(t1_dir))
        self.t2_images = sorted(os.listdir(t2_dir))
        self.labels = sorted(os.listdir(label_dir))

        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.label_dir = label_dir

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.t1_images)

    def __getitem__(self, idx):
        t1 = Image.open(os.path.join(self.t1_dir, self.t1_images[idx])).convert("RGB")
        t2 = Image.open(os.path.join(self.t2_dir, self.t2_images[idx])).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, self.labels[idx]))

        t1 = self.transform(t1)
        t2 = self.transform(t2)
        label = self.transform(label)

        return t1, t2, label