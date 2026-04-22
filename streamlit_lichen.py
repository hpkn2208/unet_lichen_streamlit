import os
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
import segmentation_models_pytorch as smp
from PIL import Image
import io
import base64

# ---------------------------------------------------------
# Developer notes / maintenance checkpoints
# - `model_path` points to the U-Net checkpoint file.
# - `classifier_path` points to the oral/non-oral classifier checkpoint.
# - The app can run without a classifier if `classifier_path` is empty
#   or the checkpoint file is missing.
# - If you change the classifier architecture, update `load_classifier`
#   and the default architecture choices here.
# ---------------------------------------------------------

st.set_page_config(page_title="Oral Lichen Segmentation Demo", layout="wide")

st.title("Oral Lichen Segmentation Demo")
st.write("Upload one or more images, run segmentation with optional oral/non-oral classification.")


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


# Segmentation model selection
st.subheader("Segmentation Settings")
segmentation_checkpoint = st.selectbox(
    "Segmentation model",
    ["UNet.pth", "UNet_plusplus.pth"],
    index=0,
    help="Select which segmentation model to use for lichen detection"
)

# Auto-detect architecture from checkpoint name
if "plusplus" in segmentation_checkpoint.lower():
    segmentation_model_type = "unetplusplus"
else:
    segmentation_model_type = "unet"

model_path = f"model/{segmentation_checkpoint}"

# Classifier settings
st.subheader("Classifier Settings")
use_classifier = st.checkbox("Use Oral/Non-oral Classifier", value=True, help="Enable pre-filtering with oral image classifier")

classifier_path = None
classifier_arch = None
if use_classifier:
    classifier_checkpoint = st.selectbox(
        "Classifier model",
        ["oral_classifier_mobilenetv3_small_100.pth", "oral_classifier_resnet18.pth", 
         "oral_classifier_resnet34.pth", "oral_classifier_resnet50.pth"],
        index=0,
        help="Select which classifier model to use"
    )
    
    # Auto-detect architecture from checkpoint name
    if "mobilenetv3" in classifier_checkpoint:
        classifier_arch = "mobilenetv3_small_100"
    elif "resnet50" in classifier_checkpoint:
        classifier_arch = "resnet50"
    elif "resnet34" in classifier_checkpoint:
        classifier_arch = "resnet34"
    elif "resnet18" in classifier_checkpoint:
        classifier_arch = "resnet18"
    
    classifier_path = f"model/{classifier_checkpoint}"

# Thresholds
st.subheader("Detection Thresholds")
col1, col2 = st.columns(2)
with col1:
    if use_classifier:
        classification_threshold = st.slider("Classification threshold", 0.0, 1.0, 0.8, 0.01)
    else:
        classification_threshold = 0.5
with col2:
    segmentation_threshold = st.slider("Segmentation threshold", 0.0, 1.0, 0.7, 0.01)

status = st.empty()


def infer_segmentation_architecture(state_dict):
    if any(key.startswith("decoder.blocks.x_0_") for key in state_dict):
        return "unetplusplus"
    if any(key.startswith("decoder.blocks.") for key in state_dict):
        return "unet"
    return None


@st.cache_resource
def load_unet(path, model_type="unet"):
    if not os.path.exists(path):
        return None, ["Checkpoint path does not exist."]

    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    classes = 1
    for key, value in state_dict.items():
        if key.endswith("segmentation_head.0.weight"):
            classes = value.shape[0]
            break

    inferred_type = infer_segmentation_architecture(state_dict)
    selected_type = inferred_type if inferred_type is not None else model_type

    if selected_type == "unet":
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=classes)
    else:
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=classes,
            decoder_attention_type="scse",
        )

    load_result = model.load_state_dict(state_dict, strict=False)
    warnings = []
    if inferred_type is not None and inferred_type != model_type:
        warnings.append(
            f"Checkpoint appears to be {inferred_type}, loading with that architecture instead of {model_type}."
        )
    if load_result.missing_keys:
        warnings.append(
            f"Missing keys in checkpoint: {load_result.missing_keys[:10]}{'...' if len(load_result.missing_keys) > 10 else ''}"
        )
    if load_result.unexpected_keys:
        warnings.append(
            f"Unexpected checkpoint keys: {load_result.unexpected_keys[:10]}{'...' if len(load_result.unexpected_keys) > 10 else ''}"
        )

    model.eval()
    return model, warnings


@st.cache_resource
def load_classifier(path, arch):
    # DEV CHECKPOINT: add new architectures here if you train a different classifier.
    if not path or not os.path.exists(path):
        return None

    if arch.startswith("mobilenetv3"):
        model = timm.create_model(arch, pretrained=False)
        # Notebook-trained MobileNetV3 model uses a classifier head with 2 outputs.
        if hasattr(model, "classifier"):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 2)
        elif hasattr(model, "head"):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, 2)
        else:
            return None
    elif hasattr(models, arch):
        model = getattr(models, arch)(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    else:
        return None

    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(img, size):
    # Preprocessing must match training normalization for both classifier and U-Net.
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
unet_model, unet_warnings = load_unet(model_path, segmentation_model_type)
classifier_model = None

if unet_model is None:
    st.warning("U-Net model checkpoint not found. Upload or place the model in the same folder.")
    st.stop()

for warning in unet_warnings:
    st.warning(warning)

st.success("U-Net checkpoint loaded successfully.")
st.markdown(
    f"**Loaded U-Net checkpoint:** `{model_path}`  \n"
    f"**Segmentation architecture:** `{'U-Net' if segmentation_model_type == 'unet' else 'U-Net++'}`"
)

if use_classifier and classifier_path and classifier_arch:
    classifier_model = load_classifier(classifier_path, classifier_arch)
    if classifier_model is not None:
        st.success("Classifier checkpoint loaded successfully.")
        st.markdown(f"**Loaded classifier:** `{classifier_path}` → `{classifier_arch}`")
    else:
        st.warning("Classifier checkpoint not found or invalid. Running segmentation only.")
        classifier_model = None
elif use_classifier:
    st.info("Classifier enabled but no model selected.")
else:
    st.info("Classifier disabled. Running U-Net on all uploaded images.")

uploaded_files = st.file_uploader("Upload PNG/JPG images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload images to run detection.")
    st.stop()

# DEV CHECKPOINT: change row_cols to control how many images are rendered per row.
uploaded_files = list(uploaded_files)[::-1]
row_cols = 4
for i in range(0, len(uploaded_files), row_cols):
    cols = st.columns(row_cols, gap="small", vertical_alignment="top", border=True)
    for j, uploaded in enumerate(uploaded_files[i:i+row_cols]):
        img = Image.open(uploaded).convert("RGB")
        arr = np.array(img)
        image_tensor = preprocess_image(img, 224)
        is_oral = True

        cls_prob = None
        if classifier_model is not None:
            with torch.no_grad():
                cls_logits = classifier_model(image_tensor)
                cls_scores = F.softmax(cls_logits, dim=1)[0].cpu().numpy()
                cls_prob = float(cls_scores[1])
            is_oral = cls_prob >= classification_threshold

        small = img.resize((256, 256))
        input_tensor = preprocess_image(small, 256)

        overlay = arr.copy()
        pred_resized = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)

        if is_oral:
            with torch.no_grad():
                logits = unet_model(input_tensor)
                if logits.shape[1] == 1:
                    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                else:
                    prob = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
            pred = (prob > segmentation_threshold).astype(np.uint8) * 255
            pred_resized = np.array(Image.fromarray(pred).resize((arr.shape[1], arr.shape[0]), Image.NEAREST))
            red_mask = np.zeros_like(overlay)
            red_mask[pred_resized == 255] = [255, 0, 0]
            overlay = (overlay * 0.8 + red_mask * 0.2).astype(np.uint8)

        col = cols[j]
        with col:
            st.markdown(f"#### {uploaded.name}")
            show_responsive_image(arr, caption="Uploaded image")

            if classifier_model is not None:
                if is_oral:
                    st.markdown(
                        f"<span style='color:blue; font-weight:bold'>Classifier: oral image ({cls_prob:.2f}) — running U-Net</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<span style='color:orange; font-weight:bold'>Classifier: not oral ({cls_prob:.2f}) — skipping U-Net</span>",
                        unsafe_allow_html=True,
                    )
                st.caption(f"Classification threshold: {classification_threshold:.2f}")

            if not is_oral:
                st.warning("Skipped U-Net: image classified as not oral.")
                continue

            if pred_resized.max() > 0:
                st.markdown('<span style="color:red; font-weight:bold">Segmentation predicted: lichen</span>', unsafe_allow_html=True)
                show_responsive_image(overlay, caption="Overlay")
            else:
                st.markdown('<span style="color:green; font-weight:bold">Segmentation predicted: no lesion</span>', unsafe_allow_html=True)
                show_responsive_image(overlay, caption="No oral lesion found")

st.success("Done. Predictions shown above.")
