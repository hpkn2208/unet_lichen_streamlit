# Oral Lichen Segmentation - Cascaded 3-Stage Pipeline

This repo contains a Streamlit app implementing a cascaded 3-stage pipeline for Oral Lichen Planus detection and segmentation:

**Stage 1:** Oral/Non-oral Binary Classifier (MobileNetV3/ResNet)  
**Stage 2:** Disease 3-Class Classifier (Normal | Lichen Planus | Other) - NEW  
**Stage 3:** Lichen Lesion Segmentation (U-Net / U-Net++)

## Overview

The app processes oral images through three stages:
1. **Stage 1 (Gate Keeper)**: Filters non-oral images using an oral/non-oral binary classifier
2. **Stage 2 (Disease Classifier)**: Classifies oral images into 3 categories (Normal healthy mucosa, Lichen Planus lesions, Other diseases)
3. **Stage 3 (Segmenter)**: Only segments images predicted as Lichen Planus, providing pixel-level lesion boundaries

This cascaded approach reduces false positives by filtering non-oral images early and only segmenting high-confidence Lichen Planus predictions.

## Files
- `streamlit_lichen.py`: Main Streamlit inference app
- `requirements.txt`: Python dependencies
- `model/`: Directory containing model checkpoints:
  - `UNet.pth`: U-Net segmentation model
  - `UNet_plusplus.pth`: U-Net++ segmentation model
  - `oral_classifier_*.pth`: Stage 1 oral/non-oral classifiers
  - `stage2_classifier.pth`: Stage 2 disease classifier (generated from `classification_stage2.ipynb`)

## Setup

1. Create the Python environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Place model checkpoints in the `model/` folder:
   - **Segmentation Models**: Copy `UNet.pth` or `UNet_plusplus.pth` to `model/`
   - **Stage 1 Classifier**: Copy any `oral_classifier_*.pth` to `model/`
   - **Stage 2 Classifier**: Run `classification_stage2.ipynb` to train and generate `stage2_classifier.pth` (automatically copies to `model/` folder)

## Run

```bash
streamlit run streamlit_lichen.py
```

The app will open in your browser, typically at `http://localhost:8501`.

## Usage

### Interface

The sidebar provides configuration options for all three stages:

**1. Segmentation Settings:**
- Select segmentation model (U-Net or U-Net++)

**2. Stage 1: Oral/Non-oral Classifier**
- Enable/disable Stage 1 filtering
- Select classifier architecture (MobileNetV3_small_100, ResNet18/34/50)
- Set classification threshold (default: 0.8)

**3. Stage 2: Disease Classifier** (NEW)
- Enable/disable Stage 2 disease classification
- Set confidence threshold for disease prediction (default: 0.7)
- Classes: Normal | Lichen Planus | Other

**4. Detection Thresholds:**
- Adjust probability thresholds for each stage

### Workflow

1. Upload one or more oral images (PNG, JPG, JPEG)
2. The app automatically processes them through all stages:
   - **Stage 1**: Determines if image is oral or non-oral
     - ✓ Blue checkmark: Image classified as oral → proceed to Stage 2
     - ✗ Orange X: Image classified as non-oral → skip further processing
   - **Stage 2**: Classifies disease category (only for oral images)
     - Red: Lichen Planus (high confidence) → proceed to segmentation
     - Green: Normal healthy mucosa → skip segmentation
     - Orange: Other disease → skip segmentation
   - **Stage 3**: Segments lesion boundaries (only for Lichen Planus)
     - Red overlay: Detected lesion regions
     - Green overlay: No lesion detected

### Output

For each image, the app displays:
1. **Original uploaded image**
2. **Stage 1 result**: Oral/Non-oral classification + confidence score
3. **Stage 2 result**: Disease category + confidence score (if Stage 1 passed)
4. **Stage 3 result**: Segmentation overlay (if Stages 1 & 2 passed)

## Model Information

### Stage 1 Classifier
- **Architecture**: MobileNetV3 or ResNet (18/34/50)
- **Input size**: 224×224 pixels
- **Output**: Binary classification (Oral/Non-oral)
- **Checkpoint format**: Direct state_dict or wrapped checkpoint

### Stage 2 Classifier
- **Architecture**: EfficientNet-B0 (pre-trained on ImageNet)
- **Input size**: 384×384 pixels
- **Output**: 3-class classification (Normal, Lichen Planus, Other)
- **Feature extraction**: 1280-dimensional features from avgpool layer
- **Checkpoint format**: Checkpoint dict with `model_state_dict`, `class_mapping`, and metadata
- **Training**: See `classification_stage2.ipynb` in the training_script folder

### Stage 3 Segmenter
- **Architecture**: U-Net or U-Net++ with ResNet34 encoder
- **Input size**: 256×256 pixels
- **Output**: Binary segmentation mask (lesion/no-lesion per pixel)
- **Checkpoint format**: State dict or wrapped checkpoint

## Preprocessing

All models use ImageNet normalization:
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Images are automatically resized to the appropriate input size for each model.

## Performance

The cascaded approach significantly reduces computational cost and false positives:
- **Stage 1 filters**: Non-oral images (photos of non-mouth surfaces)
- **Stage 2 filters**: Non-lesion oral images (healthy mucosa, other diseases)
- **Stage 3 segments**: Only high-confidence Lichen Planus cases

This reduces segmentation false positives on normal mucosa and teeth surfaces, addressing the original issue where Stage 3 alone was generating too many false positive segmentations.

## Configuration

To modify thresholds or model paths, edit the sidebar settings in the Streamlit interface. These settings persist during your session but reset on page reload.

### Advanced Configuration

Edit `streamlit_lichen.py` to:
- Change default threshold values
- Modify model search paths
- Add new classifier architectures
- Adjust preprocessing parameters
- Change visualization colors

## Troubleshooting

**"Model checkpoint not found"**
- Ensure model files are in the `model/` folder or provide full paths
- Check file extensions (.pth)

**"Stage 2 classifier not loaded"**
- Verify `stage2_classifier.pth` exists in `model/` folder
- Run `classification_stage2.ipynb` to train and generate the model
- Check that PyTorch and related dependencies are properly installed

**Predictions seem incorrect**
- Verify image preprocessing matches training data normalization
- Check threshold values are appropriate for your use case
- Inspect per-class metrics in model info files

## Notes

- All three stages can be disabled independently to test individual components
- The app uses Streamlit caching (`@st.cache_resource`) for efficient model loading
- GPU acceleration is used if CUDA is available
- Images are displayed responsively with HTML/CSS for better visualization

## References

- **Stage 1**: Oral vs Non-oral classifier (existing implementation)
- **Stage 2**: EfficientNet-B0 based disease classifier with t-SNE feature analysis
- **Stage 3**: U-Net/U-Net++ segmentation models with scSE attention
