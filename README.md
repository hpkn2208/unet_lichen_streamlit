# Oral Lichen Detection Streamlit App

This repo contains a simple Streamlit app to run lichen segmentation inference on uploaded images.

## Files
- `streamlit_lichen.py`: Streamlit inference app
- `requirements.txt`: Python dependencies
- `model.pth`: Expected model checkpoint file (place in same folder or provide path)

## Setup
1. Create environment and install:
   ```bash
   cd streamlit
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Put your trained model checkpoint (UNet) in the folder:
   - `model.pth`

## Run
```bash
streamlit run streamlit_lichen.py
```

## Usage
1. Open Streamlit URL
2. Ensure model path is set to `model.pth` or your checkpoint path
3. Upload one or more images (png/jpg)
4. View predictions and overlays

## Notes
- The model architecture uses `smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)`.
- If you use a different architecture or normalization, update `streamlit_lichen.py` accordingly.

# unet_lichen_streamlit
Deploy unet lichen detection model over streamlt
