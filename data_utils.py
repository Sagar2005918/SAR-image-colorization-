import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from skimage.color import rgb2lab

def rgb_to_lab_np(image):
    return rgb2lab(image)

def get_lab(image):
    image = np.asarray(image).astype("float32") / 255.0
    lab = rgb_to_lab_np(image)
    L = lab[:, :, 0:1] / 100.0
    ab = lab[:, :, 1:] / 128.0
    return L.transpose(2,0,1), ab.transpose(2,0,1)

def load_dataset(path, batch_size=4):
    transform = transforms.Resize((256, 256))
    images = datasets.ImageFolder(path, transform=transform)
    L_list, ab_list = [], []
    for img, _ in images:
        L, ab = get_lab(img)
        L_list.append(L)
        ab_list.append(ab)
    L_tensor = torch.tensor(np.array(L_list), dtype=torch.float32)
    ab_tensor = torch.tensor(np.array(ab_list), dtype=torch.float32)
    dataset = list(zip(L_tensor, ab_tensor))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
