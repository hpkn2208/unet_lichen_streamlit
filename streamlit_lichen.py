import os
import streamlit as st
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image

st.set_page_config(page_title="Oral Lichen Detector", layout="wide")

st.title("Oral Lichen Segmentation Demo")
st.write("Upload one or more images, then run the model to see lichen overlay predictions.")

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

out_cols = st.columns(2)

for idx, uploaded in enumerate(uploaded_files):
    img = Image.open(uploaded).convert("RGB")
    arr = np.array(img)
    inp = cv2.resize(arr, (256, 256))
    inp = inp.astype(np.float32) / 255.0
    inp = (inp - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    inp = inp.transpose(2, 0, 1)
    inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(inp_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred = (prob > 0.5).astype(np.uint8) * 255
    pred_resized = cv2.resize(pred, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = arr.copy()
    red_mask = np.zeros_like(overlay)
    red_mask[pred_resized == 255] = [255, 0, 0]
    overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

    col = out_cols[idx % 2]
    col.subheader(uploaded.name)
    col.image(arr, caption="Input image", width=400)
    caption = "Predicted lichen overlay"
    if pred_resized.max() == 0:
        caption = "Predicted normal"
    col.image(overlay, caption=caption, width=400)

st.success("Done. Predictions shown above.")
