import os
import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import io
import base64

st.set_page_config(page_title="Oral Lichen Segmentation Demo", layout="wide")

st.title("Oral Lichen Segmentation Demo")
st.write("Upload one or more images, then run the model to see lichen overlay predictions.")

def show_responsive_image(arr, caption=None):
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    html = "<div style='text-align:center; margin:4px 0;'>"
    html += f"<img src='data:image/png;base64,{data}' style='max-width:100%;height:auto;border-radius:8px;'/>"
    if caption:
        html += f"<div style='font-size:13px; color:#ccc; margin-top:4px;'>{caption}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

model_path = st.text_input("Model checkpoint path", "model.pth")

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model(model_path)

if model is None:
    st.warning("Model checkpoint not found. Upload or place model.pth in the same folder.")
    st.stop()

uploaded_files = st.file_uploader("Upload PNG/JPG images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload images to run detection.")
    st.stop()

# Most recent uploads first
uploaded_files = list(uploaded_files)[::-1]

# display in grid with uniform spacing
row_cols = 4
for i in range(0, len(uploaded_files), row_cols):
    cols = st.columns(row_cols, gap="small", vertical_alignment="top", border=True)
    for j, uploaded in enumerate(uploaded_files[i:i+row_cols]):
        img = Image.open(uploaded).convert("RGB")
        arr = np.array(img)
        small = img.resize((256, 256))
        inp = np.array(small).astype(np.float32) / 255.0
        inp = (inp - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        inp = inp.transpose(2, 0, 1)
        inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(inp_tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8) * 255
        pred_resized = np.array(Image.fromarray(pred).resize((arr.shape[1], arr.shape[0]), Image.NEAREST))

        overlay = arr.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[pred_resized == 255] = [255, 0, 0]
        overlay = (overlay * 0.8 + red_mask * 0.2).astype(np.uint8)

        col = cols[j]
        with col:
            st.markdown(f"#### {uploaded.name}")
            show_responsive_image(arr, caption="Uploaded image")
            if pred_resized.max() > 0:
                st.markdown('<span style="color:red; font-weight:bold">Model predicted: lichen</span>', unsafe_allow_html=True)
                show_responsive_image(overlay, caption="Overlay")
            else:
                st.markdown('<span style="color:green; font-weight:bold">Model predicted: normal</span>', unsafe_allow_html=True)
                show_responsive_image(overlay, caption="No oral lesion found")

st.success("Done. Predictions shown above.")
