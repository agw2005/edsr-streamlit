
# '''
# Untuk menjalankan aplikasi streamlit secara local, lakukan instalasi modul streamlit melalui command prompt dengan perintah
# `pip install streamlit`, kemudian setelah berhasil terinstall aplikasi dapat berjalan dengan mengetikkan perintah
# `streamlit run app.py` pada tempat dimana kamu menyimpan file app.py milikmu. Jangan lupa tambahkan file requirements juga
# yang berisi library python yang dipakai agar aplikasi bisa berjalan.
# '''

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# from image_utils import *

# st.set_page_config("Super Resolution Application using EDSR Method", layout="wide")
# st.title("🖼️ Super-Resolution Application")
# st.markdown("Unggah gambar dan pilih metode pengolahan citra dari menu di sebelah kiri.")

# st.sidebar.title("🔧 Pilih Metode")
# method = st.sidebar.selectbox("Metode Pengolahan", [
#     "Grayscale",
#     "Gaussian Blur",
#     "Otsu Thresholding",
#     "Prewitt Edge Detection",
#     "Sobel Edge Detection",
#     "Histogram Equalization",
#     "Quantizing Compression"
# ])

# uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")

# else:
#     st.warning("⚠️ Harap unggah gambar terlebih dahulu.")
