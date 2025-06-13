import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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

def edsr_max_infer(image_input, model_path, device='cuda', show=False, save_path=None):
    # Perform inference with a trained EDSR model
    # Arguments :
    # # image_input (str or PIL.Image): Path to image or PIL Image
    # # model_path (str): Path to .pth model weights (Assumed to be a state dictionary)
    # # device (str): Device to use ('cuda' or 'cpu')
    # # show (bool): If True, display input vs output images (and also its respective histogram)
    # # save_path (str): If provided, save the result image to this path
    # Return :
    # # PIL.Image: The high-resolution output image.
    
    # Load the model
    model = EDSR().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be a file path or PIL.Image")
    
    # Transform input
    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Clamp and convert output tensor to image
    output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
    output_image = transforms.ToPILImage()(output_tensor)

    if show:
        # === First figure: image comparison ===
        fig_img, axes_img = plt.subplots(1, 2, figsize=(12, 8))
        axes_img[0].imshow(image)
        axes_img[0].set_title("Input (Low Resolution)")
        axes_img[0].axis("off")

        axes_img[1].imshow(output_image)
        axes_img[1].set_title("Output (Super Resolution)")
        axes_img[1].axis("off")

        plt.tight_layout()
        plt.show()

        # === Second figure: histogram comparison ===
        fig_hist, axes_hist = plt.subplots(1, 2, figsize=(13, 5))

        def plot_hist(ax, img, title):
            img_np = np.array(img)
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = np.histogram(img_np[:, :, i], bins=256, range=(0, 255))[0]
                ax.plot(hist, color=color, label=color)
            ax.set_title(title)
            ax.set_xlim([0, 255])
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            ax.legend()

        plot_hist(axes_hist[0], image, "Input Histogram")
        plot_hist(axes_hist[1], output_image, "Output Histogram")

        plt.tight_layout()
        plt.show()

    if save_path:
        output_image.save(save_path)

    return output_image
