import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gc
import streamlit as st

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out * 0.1 + residual  # residual scaling
    
class EDSR(nn.Module):
    def __init__(self, scale_factor=2, num_channels=64, num_blocks=16):
        super().__init__()
        self.entry = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.mid_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.entry(x)
        res = self.res_blocks(x)
        x = self.mid_conv(res) + x
        x = self.upsample(x)
        return x

def edsr_max_infer(image_input, model, device='cpu', show=False, save_path=None):
    model.eval()
    
    # Load image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be a file path or PIL.Image")
    
    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu().clamp(0, 1)

    # Cleanup
    del input_tensor
    gc.collect()

    output_image = transforms.ToPILImage()(output_tensor)

    if save_path:
        output_image.save(save_path)

    return output_image

def run_inference_and_cleanup(image, model_path):
    device = 'cpu'
    model = EDSR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu().clamp(0, 1)

    # Cleanup
    del model
    del input_tensor
    torch.cuda.empty_cache()
    gc.collect()

    return transforms.ToPILImage()(output_tensor)

@st.cache_resource
def load_model(model_path):
    device = 'cpu'
    model = EDSR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model