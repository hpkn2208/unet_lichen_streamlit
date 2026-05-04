import streamlit as st
import numpy as np
from PIL import Image
import json
import zipfile
from datetime import datetime
from pathlib import Path


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