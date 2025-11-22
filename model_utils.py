import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage.color import lab2rgb
from data_utils import get_lab



def build_model():
    model = nn.Sequential(
        nn.Conv2d(1, 64, 3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 256, 3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1),
        nn.Tanh()
    )
    return model



def lab_to_rgb(L, ab):
    
    if L.ndim == 2:
        L = L[np.newaxis, :, :]    
    elif L.ndim == 3 and L.shape[0] != 1:
        L = L[0:1, :, :]           

    
    if ab.ndim == 4:  
        ab = ab[0]  

   
    Lab = np.concatenate([L, ab], axis=0).transpose(1, 2, 0)

    
    Lab[:, :, 0] *= 100
    Lab[:, :, 1:] *= 128

    rgb = (lab2rgb(Lab) * 255).astype(np.uint8)
    return Image.fromarray(rgb)




def predict_image(model, grayscale_path, output="output.png"):
    img = Image.open(grayscale_path).convert("RGB").resize((256, 256))

    
    L, _ = get_lab(img)  

    L_tensor = torch.tensor(L).unsqueeze(0).float()  

    with torch.no_grad():
        ab = model(L_tensor)[0].detach().cpu().numpy()  

    img_out = lab_to_rgb(L, ab)
    img_out.save(output)

    print("Saved:", output)
