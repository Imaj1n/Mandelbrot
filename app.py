import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, float32, int32
from PIL import Image

# st.set_page_config(layout="wide", page_title="Mandelbrot Explorer")

# === Numba-accelerated Mandelbrot ===
@njit
def mandelbrot(c_real: float32, c_imag: float32, max_iter: int32) -> int32:
    z_real = float32(0.0)
    z_imag = float32(0.0)
    for i in range(max_iter):
        z_real2 = z_real * z_real
        z_imag2 = z_imag * z_imag
        if z_real2 + z_imag2 > 4.0:
            return i
        z_imag = 2 * z_real * z_imag + c_imag
        z_real = z_real2 - z_imag2 + c_real
    return max_iter

@njit(parallel=True)
def create_mandelbrot(width: int32, height: int32,
                      re_min: float32, re_max: float32,
                      im_min: float32, im_max: float32,
                      max_iter: int32):
    image = np.zeros((height, width), dtype=np.int32)
    for y in prange(height):
        im = float32(im_min + y * (im_max - im_min) / height)
        for x in range(width):
            re = float32(re_min + x * (re_max - re_min) / width)
            image[y, x] = mandelbrot(re, im, max_iter)
    return image

# === Sidebar Inputs ===
st.sidebar.title("Mandelbrot Parameters")

re_center = float32(st.sidebar.number_input("Real Center", value=-0.745, format="%.6f"))
im_center = float32(st.sidebar.number_input("Imaginary Center", value=0.112, format="%.6f"))
zoom_width = float32(st.sidebar.number_input("Zoom Width", min_value=0.00001, value=0.01, format="%.6f"))
st.sidebar.caption("Nilai zoom 4.00 memperlihatkan keseluruhan set Fraktal Mandelbrot")
max_iter = int32(st.sidebar.slider("Max Iterations", 100, 2000, 1000, step=100))
SS = int32(st.sidebar.slider("Supersampling", 1, 4, 2))

# === Area Calculations ===
re_min = float32(re_center - zoom_width / 2)
re_max = float32(re_center + zoom_width / 2)
im_min = float32(im_center - zoom_width / 2)
im_max = float32(im_center + zoom_width / 2)

width = int32(1080)
height = int32(int(width * (im_max - im_min) / (re_max - re_min)))

ss_width = width * SS
ss_height = height * SS

# === Fractal Computation ===
with st.spinner("⏳ Generating Mandelbrot set..."):
    image_hi_res = create_mandelbrot(ss_width, ss_height, re_min, re_max, im_min, im_max, max_iter)

# === Antialiasing via Supersampling ===
img_pil = Image.fromarray(image_hi_res.astype(np.uint16))
img_resized = img_pil.resize((width, height), resample=Image.LANCZOS)
image = np.array(img_resized)

# === Normalization for Coloring ===
image_normalized = np.log(image + 1) / np.log(max_iter)

# === Colormap Options ===
st.subheader("🎨 Pilihan Colormap")
colormaps = [
    "twilight_shifted", "magma", "inferno", "plasma", "viridis",
    "cividis", "cubehelix", "hot", "cool", "spring", "summer", "autumn", "winter", "turbo"
]
selected_cmap = st.selectbox("Choose a colormap", colormaps, index=0)

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_normalized, cmap=selected_cmap, extent=[re_min, re_max, im_min, im_max])
ax.axis('off')
plt.tight_layout(pad=0)
st.pyplot(fig)
