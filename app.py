import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from image_utils import *

st.set_page_config("Super Resolver", layout="wide")
st.title("🖼️ Super-Resolution Application using EDSR Method 🤖")
st.markdown("Upload an image as input and output a super-resolved version of that image (png) using EDSR")
st.markdown("This implementation is a simplified version of the Enhanced Deep Super-Resolution (EDSR) model as described in the paper \"Enhanced Deep Residual Networks for Single Image Super-Resolution\" (Lim et al., 2017).")
st.markdown("If the input is a gif and webp, the model will only process the first frame")

uploaded_file = st.file_uploader("Upload Your Image ⬇️", type=["jpg", "jpeg", "png", "bmp", "tiff", "gif", "webp"])
model = "best_edsr.pth"

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image (Low Resolution)", use_container_width=True)

    # Run inference
    with st.spinner("Running super-resolution inference... (This might take up to 2 minutes depending on the size of the image)"):
        model_path = model  # adjust if different
        device = "cpu"  # since you assume no CUDA device
        output_image = edsr_max_infer(image, model_path=model_path, device=device)

    st.success("Inference complete! Here's the result:")

    # Show output image
    st.image(output_image, caption="Output Image (Super Resolution)", use_container_width=True)

    # Save output to buffer for download
    from io import BytesIO
    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Download button
    st.download_button(
        label="Download Super-Resolved Image ⬇️",
        data=buffer,
        file_name="super_res_output.png",
        mime="image/png"
    )


else:
    st.warning("Waiting for an image 😴")
