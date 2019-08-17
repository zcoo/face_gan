from torch.utils import data
import os, sys

import torch
from PIL import Image
import torchvision.transforms as tvt

image_trans = tvt.Compose([
    tvt.Resize((128, 128)),
    tvt.ToTensor(),
    #tvt.Normalize(mean=[0.485, 0.456, 0.406],
    #              std=[0.229, 0.224, 0.225])
])

class Dataset():
    """
    Dataset for training/testing the face GAN network
    """

    def __init__(self, data_dirs):
        self.data_dir = data_dirs
        self.data=[]
        for data_dir in data_dirs:
            self.data += [data_dir+"/"+ tmp for tmp in os.listdir(data_dir)]
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        raw_data = self.data[index]
        raw_data = image_trans(Image.open(raw_data).convert('RGB'))
        return raw_data


if __name__ == "__main__":
    data_dirs=["./MTFL/AFLW","./MTFL/lfw_5590","./MTFL/net_7876"]
    ds = Dataset(data_dirs)
    print(ds[0])
