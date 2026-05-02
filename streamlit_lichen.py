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
import json
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------
# Force CPU mode to avoid GPU memory issues
# Set to 'cuda' if you have sufficient GPU memory and paging file
# ---------------------------------------------------------
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

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


def create_feedback_zip():
    """Create a zip file of all feedback data"""
    feedback_dir = Path("feedback_data")
    
    if not feedback_dir.exists() or not any(feedback_dir.glob('**/*')):
        return None
    
    zip_path = Path("feedback_data.zip")
    
    # Remove existing zip if present
    if zip_path.exists():
        zip_path.unlink()
    
    # Create zip with all feedback data
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in feedback_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(feedback_dir.parent)
                zipf.write(file_path, arcname)
    
    return zip_path


# # =================================================================
# # SIDEBAR: Feedback Data Download
# # =================================================================
# with st.sidebar:
#     st.header("📊 Feedback Management")
    
#     if st.button("📥 Download Feedback Data (ZIP)", use_container_width=True, help="Download all collected feedback images and metadata"):
#         zip_path = create_feedback_zip()
        
#         if zip_path and zip_path.exists():
#             with open(zip_path, 'rb') as f:
#                 st.download_button(
#                     label="⬇️ Click to download feedback_data.zip",
#                     data=f.read(),
#                     file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
#                     mime="application/zip",
#                     use_container_width=True
#                 )
            
#             # Show feedback stats
#             feedback_dir = Path("feedback_data")
#             if feedback_dir.exists():
#                 total_files = sum(1 for _ in feedback_dir.rglob('*') if _.is_file())
#                 st.success(f"✓ {total_files} feedback items ready for download")
#         else:
#             st.info("No feedback data collected yet.")
    
#     # st.divider()




def get_image_id(filename):
    """Generate unique image ID from filename."""
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def resolve_folder(feedback_type, reason=None):
    """
    Resolve target folder based on feedback type and reason.
    
    Returns:
        str: Path to target folder
    """
    base_path = Path("feedback_data")
    
    if feedback_type == "Correct":
        folder = base_path / "Success_Data"
    elif feedback_type == "Incorrect":
        if reason == "Not mouth image":
            folder = base_path / "Stage1_Hard_Negative"
        elif reason == "Wrong disease":
            folder = base_path / "Stage2_Hard_Negative"
        elif reason == "Bad mask":
            folder = base_path / "Stage3_Hard_Negative"
        else:
            folder = base_path / "General_Feedback"
    else:
        folder = base_path / "General_Feedback"
    
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_feedback_image(image_array, image_id, feedback_type, reason=None, original_filename=None):
    """
    Save image to appropriate feedback folder.
    
    Args:
        image_array: numpy array of image
        image_id: unique image identifier
        feedback_type: "Correct" or "Incorrect"
        reason: reason for incorrect feedback (if applicable)
        original_filename: original filename for reference
    
    Returns:
        str: Path where image was saved
    """
    folder = resolve_folder(feedback_type, reason)
    
    # Create filename: image_id_timestamp.png
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_id}_{timestamp}.png"
    filepath = folder / filename
    
    # Save image
    pil_image = Image.fromarray(image_array.astype(np.uint8))
    pil_image.save(filepath)
    
    return str(filepath)


def save_feedback_metadata(metadata):
    """
    Save feedback metadata as JSON.
    
    Args:
        metadata: dict containing feedback info
            {
                'image_id': str,
                'original_filename': str,
                'prediction': dict,
                'feedback': str,
                'reason': str or None,
                'correct_class': str or None,
                'timestamp': str,
                'model_version': str,
                'image_path': str
            }
    """
    folder = resolve_folder(metadata.get("feedback"), metadata.get("reason"))
    metadata_file = folder / f"{metadata['image_id']}_metadata.json"
    
    # Ensure prediction dict is JSON serializable
    safe_metadata = {
        'image_id': metadata['image_id'],
        'original_filename': metadata.get('original_filename'),
        'prediction': {
            'stage1_oral': float(metadata['prediction'].get('stage1_prob', 0)),
            'stage2_disease': metadata['prediction'].get('stage2_disease_name', 'N/A'),
            'stage2_confidence': float(metadata['prediction'].get('stage2_prob', 0)),
            'stage3_segmented': metadata['prediction'].get('stage3_segmented', False),
        },
        'feedback': metadata.get('feedback'),
        'reason': metadata.get('reason'),
        'correct_class': metadata.get('correct_class'),
        'timestamp': metadata.get('timestamp'),
        'model_version': metadata.get('model_version', '1.0'),
        'models_used': metadata.get('models_used', {}),
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(safe_metadata, f, indent=2)
    
    return str(metadata_file)


def render_feedback_widget(col, image_array, image_id, predictions, uploaded_filename, models_used=None):
    """
    Render feedback widget for an image card.
    
    Args:
        col: Streamlit column object
        image_array: numpy array of original image
        image_id: unique image identifier
        predictions: dict with prediction info
            {
                'stage1_prob': float,
                'stage2_disease_name': str,
                'stage2_prob': float,
                'stage3_segmented': bool
            }
        uploaded_filename: original uploaded filename
        models_used: dict with model filenames used
            {
                'stage1_model': str,
                'stage2_model': str,
                'stage3_model': str
            }
    """
    with col:
        st.divider()
        st.markdown("📋 Feedback")
        
        # Initialize session state for this image
        feedback_key = f"feedback_{image_id}"
        reason_key = f"reason_{image_id}"
        correct_class_key = f"correct_class_{image_id}"
        submit_key = f"submit_{image_id}"
        submitted_key = f"submitted_{image_id}"
        
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = "Correct"
        if reason_key not in st.session_state:
            st.session_state[reason_key] = None
        if correct_class_key not in st.session_state:
            st.session_state[correct_class_key] = None
        if submitted_key not in st.session_state:
            st.session_state[submitted_key] = False
        
        # Check if feedback already submitted for this image
        if st.session_state[submitted_key]:
            st.info(f"✓ Thank you! Your feedback has been recorded.")
            return
        
        # Main feedback radio
        feedback = st.radio(
            "Is the prediction correct?",
            ["Correct", "Incorrect"],
            key=feedback_key,
            horizontal=True,
            index=0 if st.session_state[feedback_key] == "Correct" else 1
        )
        
        # Show reason dropdown only if Incorrect
        if feedback == "Incorrect":
            st.radio(
                "What was incorrect?",
                ["Not mouth image", "Wrong disease", "Bad mask"],
                key=reason_key,
                index=0 if st.session_state[reason_key] is None else 
                      ["Not mouth image", "Wrong disease", "Bad mask"].index(st.session_state[reason_key])
            )
            
            # Show correct class selector if "Wrong disease" is selected
            current_reason = st.session_state.get(reason_key)
            if current_reason == "Wrong disease":
                st.selectbox(
                    "What should the correct class be?",
                    ["Lichen", "Normal", "Other"],
                    key=correct_class_key,
                    index=0 if st.session_state[correct_class_key] is None else
                          ["Lichen", "Normal", "Other"].index(st.session_state[correct_class_key])
                )
        
        # Determine whether all required feedback fields are filled
        current_feedback = st.session_state[feedback_key]
        current_reason = st.session_state.get(reason_key)
        current_correct_class = st.session_state.get(correct_class_key)
        can_submit = False
        submit_help = "Select the required feedback options to enable submit."

        if current_feedback == "Correct":
            can_submit = True
        elif current_feedback == "Incorrect":
            if current_reason == "Wrong disease":
                can_submit = bool(current_correct_class)
                submit_help = "Select the correct class before submitting."
            else:
                can_submit = bool(current_reason)
                submit_help = "Select the reason for incorrect feedback before submitting."

        if can_submit:
            if st.button("Submit Feedback", key=submit_key, use_container_width=True):
                try:
                    # Save image to appropriate folder
                    image_path = save_feedback_image(
                        image_array,
                        image_id,
                        current_feedback,
                        current_reason,
                        uploaded_filename
                    )
                    
                    # Prepare metadata
                    metadata = {
                        'image_id': image_id,
                        'original_filename': uploaded_filename,
                        'prediction': {
                            'stage1_prob': predictions.get('stage1_prob'),
                            'stage2_disease_name': predictions.get('stage2_disease_name'),
                            'stage2_prob': predictions.get('stage2_prob'),
                            'stage3_segmented': predictions.get('stage3_segmented'),
                        },
                        'feedback': current_feedback,
                        'reason': current_reason,
                        'correct_class': current_correct_class,
                        'timestamp': datetime.now().isoformat(),
                        'model_version': models_used.get('stage3_model') if models_used else '1.0',
                        'models_used': models_used if models_used else {},
                    }
                    
                    # Save metadata JSON
                    metadata_path = save_feedback_metadata(metadata)
                    
                    # Mark as submitted
                    st.session_state[submitted_key] = True
                    
                    # Show success message
                    st.success(
                        f"✓ Feedback saved!\n\n"
                        f"📁 Image: `{image_path}`\n\n"
                        f"📄 Metadata: `{metadata_path}`"
                    )
                    
                    # Rerun to show thank you message
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error saving feedback: {str(e)}")
        else:
            st.info(submit_help)

st.subheader("Segmentation Settings")
segmentation_checkpoint = st.selectbox(
    "Segmentation model",
    ["UNet.pth", "UNet_plusplus.pth" ],
    index=1,
    help="Select which segmentation model to use for lichen detection"
)

# Auto-detect architecture from checkpoint name
if "plusplus" in segmentation_checkpoint.lower():
    segmentation_model_type = "unetplusplus"
else:
    segmentation_model_type = "unet"

model_path = f"model/{segmentation_checkpoint}"

# Classifier settings - STAGE 1 (Oral/Non-oral)
st.subheader("Stage 1: Oral/Non-oral Classifier")
use_stage1_classifier = st.checkbox("Enable Stage 1 (Oral/Non-oral filter)", value=True, help="Enable pre-filtering with oral image classifier")

classifier_path = None
classifier_arch = None
classifier_checkpoint = None
if use_stage1_classifier:
    classifier_checkpoint = st.selectbox(
        "Stage 1 Classifier model",
        ["oral_classifier_mobilenetv3_small_100.pth", "oral_classifier_resnet18.pth", 
         "oral_classifier_resnet34.pth", "oral_classifier_resnet50.pth"],
        index=3,
        help="Select which classifier model to use for oral/non-oral classification"
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

# Classifier settings - STAGE 2 (Disease Classification)
st.subheader("Stage 2: Disease Classifier (Normal | Lichen Planus | Other)")
use_stage2_classifier = st.checkbox("Enable Stage 2 (Disease classification)", value=True, help="Enable disease classification (Normal/Lichen Planus/Other)")

stage2_classifier_path = None
stage2_checkpoint = None
if use_stage2_classifier:
    stage2_checkpoint = st.selectbox(
        "Stage 2 Classifier model",
        ["stage2_classifier_v2.pth", "stage2_classifier_v3.pth"],
        index=1,
        help="Select Stage 2 disease classifier model"
    )
    stage2_classifier_path = f"model/{stage2_checkpoint}"

# Thresholds
st.subheader("Detection Thresholds")
col1, col2, col3 = st.columns(3)
with col1:
    if use_stage1_classifier:
        stage1_threshold = st.slider("Stage 1 threshold", 0.0, 1.0, 0.8, 0.01, help="Confidence threshold for oral classification")
    else:
        stage1_threshold = 0.5
with col2:
    if use_stage2_classifier:
        stage2_threshold = st.slider("Stage 2 threshold", 0.0, 1.0, 0.7, 0.01, help="Confidence threshold for disease classification")
    else:
        stage2_threshold = 0.5
with col3:
    segmentation_threshold = st.slider("Stage 3 threshold", 0.0, 1.0, 0.7, 0.01, help="Confidence threshold for segmentation")

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

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------
# Stage 2 Disease Classifier (Normal | Lichen Planus | Other)
# ---------------------------------------------------------

class FeatureExtractor(nn.Module):
    """
    EfficientNet-B0 with feature extraction hook
    Extracts features from penultimate layer for t-SNE visualization
    """
    def __init__(self, num_classes=3):
        super(FeatureExtractor, self).__init__()
        
        # Load pre-trained EfficientNet-B0 with proper weights parameter (deprecated 'pretrained' parameter)
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Store features from penultimate layer
        self.features = None
        
        # Register forward hook to capture penultimate layer outputs
        self.model.avgpool.register_forward_hook(self._hook_fn)
        
        # Modify classification head for 3 classes
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture features from penultimate layer"""
        self.features = output
    
    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_stage2_classifier(path):
    """Load the Stage 2 disease classifier (Normal | Lichen Planus | Other)"""
    if not path or not os.path.exists(path):
        return None, None
    
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        
        model = FeatureExtractor(num_classes=3)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Extract class mapping if available
        class_mapping = None
        if "class_mapping" in checkpoint:
            class_mapping = checkpoint["class_mapping"]
        else:
            class_mapping = {0: "Normal", 1: "Lichen Planus", 2: "Other"}
        
        return model, class_mapping
    except Exception as e:
        st.warning(f"Error loading Stage 2 classifier: {e}")
        return None, None


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
stage1_classifier_model = None
stage2_classifier_model = None
stage2_class_mapping = None

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

if use_stage1_classifier and classifier_path and classifier_arch:
    stage1_classifier_model = load_classifier(classifier_path, classifier_arch)
    if stage1_classifier_model is not None:
        st.success("Stage 1 classifier loaded successfully.")
        st.markdown(f"**Stage 1 (Oral/Non-oral):** `{classifier_path}` → `{classifier_arch}`")
    else:
        st.warning("Stage 1 classifier checkpoint not found or invalid.")
        stage1_classifier_model = None
elif use_stage1_classifier:
    st.info("Stage 1 enabled but no model selected.")
else:
    st.info("Stage 1 disabled. Running on all uploaded images.")

if use_stage2_classifier and stage2_classifier_path:
    stage2_classifier_model, stage2_class_mapping = load_stage2_classifier(stage2_classifier_path)
    if stage2_classifier_model is not None:
        st.success("Stage 2 disease classifier loaded successfully.")
        st.markdown(f"**Stage 2 (Disease Classification):** `{stage2_classifier_path}`  \nClasses: Normal | Lichen Planus | Other")
    else:
        st.warning("Stage 2 classifier checkpoint not found. Running Stage 1 + segmentation only.")
        stage2_classifier_model = None
elif use_stage2_classifier:
    st.info("Stage 2 enabled but no model found.")
else:
    st.info("Stage 2 disabled.")

# =================================================================
# Feedback Data Download
# =================================================================
st.header("📊 Feedback Management")
st.info("Click download after finishing all feedback submission to get a ZIP of all feedback data (images + json) organized by category.")
    
if st.button("📥 Download Feedback Data (ZIP)", use_container_width=True, help="Download all collected feedback images and metadata"):
    zip_path = create_feedback_zip()
    
    if zip_path and zip_path.exists():
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="⬇️ Click to download feedback_data.zip",
                data=f.read(),
                file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        # Show feedback stats
        feedback_dir = Path("feedback_data")
        if feedback_dir.exists():
            total_files = sum(1 for _ in feedback_dir.rglob('*') if _.is_file())
            st.success(f"✓ {total_files} feedback items ready for download")
    else:
        st.info("No feedback data collected yet.")

st.divider()
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
        
        # Generate unique image ID
        image_id = get_image_id(uploaded.name)
        
        image_tensor = preprocess_image(img, 224)
        
        # ============================================================
        # STAGE 1: Oral/Non-oral Classification
        # ============================================================
        is_oral = True
        stage1_prob = None
        
        if stage1_classifier_model is not None:
            with torch.no_grad():
                cls_logits = stage1_classifier_model(image_tensor)
                cls_scores = F.softmax(cls_logits, dim=1)[0].cpu().numpy()
                stage1_prob = float(cls_scores[1])
            is_oral = stage1_prob >= stage1_threshold
        
        # ============================================================
        # STAGE 2: Disease Classification (Normal | Lichen Planus | Other)
        # ============================================================
        stage2_pred = None
        stage2_prob = None
        stage2_disease_name = None
        
        if stage2_classifier_model is not None and is_oral:
            with torch.no_grad():
                disease_logits = stage2_classifier_model(image_tensor)
                disease_scores = F.softmax(disease_logits, dim=1)[0].cpu().numpy()
                stage2_pred = int(torch.argmax(disease_logits[0], dim=0).cpu())
                stage2_prob = float(disease_scores[stage2_pred])
                stage2_disease_name = stage2_class_mapping.get(stage2_pred, f"Class {stage2_pred}")
        
        # ============================================================
        # STAGE 3: Segmentation (only for Lichen Planus predictions)
        # ============================================================
        small = img.resize((256, 256))
        input_tensor = preprocess_image(small, 256)

        overlay = arr.copy()
        pred_resized = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
        stage3_segmented = False
        
        # Only segment if: (1) is oral, (2) Stage 2 predicts Lichen Planus or Stage 2 disabled
        should_segment = is_oral and (
            stage2_classifier_model is None or 
            (stage2_pred == 1 and stage2_prob >= stage2_threshold)  # 1 = Lichen Planus
        )

        col = cols[j]
        with col:
            st.markdown(f"#### {uploaded.name}")
            show_responsive_image(arr, caption="Uploaded image")
            
            # Show Stage 1 result
            if stage1_classifier_model is not None:
                if is_oral:
                    st.markdown(
                        f"<span style='color:blue; font-weight:bold'>✓ Stage 1: Oral ({stage1_prob:.2f})</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<span style='color:orange; font-weight:bold'>✗ Stage 1: Non-oral ({stage1_prob:.2f})</span>",
                        unsafe_allow_html=True,
                    )
            
            # Show Stage 2 result
            if stage2_classifier_model is not None and is_oral:
                color_map = {"Normal": "green", "Lichen Planus": "red", "Other": "orange"}
                color = color_map.get(stage2_disease_name, "gray")
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold'>→ Stage 2: {stage2_disease_name} ({stage2_prob:.2f})</span>",
                    unsafe_allow_html=True,
                )
                if stage2_pred == 1 and stage2_prob >= stage2_threshold:
                    st.markdown(f"<span style='color:red; font-weight:bold'>  ✓ Proceed to segmentation</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:gray; font-weight:bold'>  ✗ Skip segmentation</span>", unsafe_allow_html=True)
            
            # Display segmentation results if applicable
            if is_oral:
                if should_segment:
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
                    
                    if pred_resized.max() > 0:
                        stage3_segmented = True
                        st.markdown('<span style="color:red; font-weight:bold">→ Stage 3: Lesion detected</span>', unsafe_allow_html=True)
                        show_responsive_image(overlay, caption="Overlay")
                    else:
                        st.markdown('<span style="color:green; font-weight:bold">→ Stage 3: No lesion</span>', unsafe_allow_html=True)
                        show_responsive_image(overlay, caption="No lesion found")
                else:
                    if stage2_classifier_model is not None:
                        st.warning(f"Skipped segmentation: classified as {stage2_disease_name} (Stage 2).")
            else:
                st.warning("Skipped: classified as non-oral (Stage 1).")
            
            # Render feedback widget
            predictions = {
                'stage1_prob': stage1_prob,
                'stage2_disease_name': stage2_disease_name,
                'stage2_prob': stage2_prob,
                'stage3_segmented': stage3_segmented,
            }
            models_used = {
                'stage1_model': classifier_checkpoint if use_stage1_classifier else None,
                'stage2_model': stage2_checkpoint if use_stage2_classifier else None,
                'stage3_model': segmentation_checkpoint,
            }
            render_feedback_widget(col, arr, image_id, predictions, uploaded.name, models_used)

st.success("Done. Predictions shown above.")
