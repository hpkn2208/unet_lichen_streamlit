# Oral Lichen Detection Streamlit App

This repo contains a Streamlit app for lichen segmentation inference with an optional oral-image classifier.

## Files
- `streamlit_lichen.py`: Streamlit inference app
- `requirements.txt`: Python dependencies
- `model.pth`: Expected U-Net model checkpoint file (place in same folder or provide path)
- `oral_classifier.pth`: Optional binary oral-image classifier checkpoint saved from `classification.ipynb`

## Setup
1. Create the Python environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Place your model checkpoints in the folder or provide explicit paths in the app.
   - U-Net checkpoint: `model.pth`
   - Optional classifier checkpoint: `oral_classifier.pth`

## Run
```bash
streamlit run streamlit_lichen.py
```

## Usage
1. Open the Streamlit URL shown after launch.
2. Set the U-Net checkpoint path, for example `model.pth`.
3. Optionally set the oral classifier checkpoint path and choose the classifier architecture.
4. Adjust the classification and segmentation thresholds if needed.
5. Upload one or more images (`png`, `jpg`, `jpeg`).
6. The app logs whether each image is classified as oral or not.
   - If classified as oral, U-Net segmentation runs and a red overlay is shown for predicted lichen.
   - If classified as not oral, U-Net is skipped and a clear warning is displayed.

## Notes
- The U-Net architecture uses `smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)`.
- The optional classifier uses a torchvision ResNet with a single output logit and sigmoid activation.
- If no classifier is provided, the app will run U-Net on all uploaded images.
- If you use a different architecture or normalization, update `streamlit_lichen.py` accordingly.
