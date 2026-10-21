
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from test_degradation_estimator import predict_image
from image_enhancement import enhance_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WhatsProof | Image Analysis",
    page_icon="🖼️",
    layout="wide",
)

st.title("WhatsProof")
st.subheader("WhatsApp Image Degradation Analysis")

st.write(
    "Upload an image to estimate whether it is an original or "
    "forwarded image, examine the estimated degradation indicator, "
    "and compare the image before and after enhancement."
)

st.info(
    "The degradation indicator is a dataset-relative heuristic. "
    "It is not a literal percentage of pixels or image quality lost."
)


# ============================================================
# MODEL FILE CHECK
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "degradation_estimator_model.joblib"
SCHEMA_FILE = ROOT / "degradation_estimator_features.json"

if not MODEL_FILE.is_file():
    st.error(
        "The trained degradation model is missing. "
        "Please ensure degradation_estimator_model.joblib "
        "is uploaded to the GitHub repository."
    )
    st.stop()

if not SCHEMA_FILE.is_file():
    st.error(
        "The feature schema is missing. "
        "Please ensure degradation_estimator_features.json "
        "is uploaded to the GitHub repository."
    )
    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, and WEBP.",
)

if uploaded_file is not None:

    original_bytes = uploaded_file.getvalue()

    try:
        # Validate and display the uploaded image.
        with Image.open(uploaded_file) as image:
            image.verify()

        st.success("Image uploaded successfully.")

        st.subheader("Uploaded Image")

        st.image(
            original_bytes,
            caption=uploaded_file.name,
            width="stretch",
        )

    except Exception:
        st.error(
            "This file could not be opened as a valid image. "
            "Please upload another JPG, JPEG, PNG, or WEBP image."
        )
        st.stop()

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze_clicked = st.button(
        "Analyze and Enhance Image",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:

        with st.spinner(
            "Estimating degradation and enhancing the image..."
        ):

            try:
                # Temporary files are removed after processing.
                # The uploaded original is never overwritten.
                suffix = Path(uploaded_file.name).suffix.lower()

                if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                    suffix = ".png"

                with tempfile.TemporaryDirectory(
                    prefix="whatsproof_"
                ) as temporary_directory:

                    temp_dir = Path(temporary_directory)

                    original_path = temp_dir / f"uploaded{suffix}"
                    enhanced_path = temp_dir / "enhanced.png"

                    # Save the upload temporarily so both existing
                    # Python modules can process it by filesystem path.
                    original_path.write_bytes(original_bytes)

                    # 1. Run the existing degradation estimator.
                    result = predict_image(
                        original_path,
                        save_preview=False,
                    )

                    # 2. Run the existing enhancement pipeline.
                    enhance_image(
                        original_path,
                        enhanced_path,
                        show_comparison=False,
                    )

                    # Read enhanced bytes before temporary cleanup.
                    enhanced_bytes = enhanced_path.read_bytes()

                # Save only result data and image bytes in this
                # Streamlit session, not permanent server-side files.
                st.session_state["whatsproof_result"] = result
                st.session_state["whatsproof_original_bytes"] = (
                    original_bytes
                )
                st.session_state["whatsproof_enhanced_bytes"] = (
                    enhanced_bytes
                )
                st.session_state["whatsproof_filename"] = (
                    uploaded_file.name
                )

            except Exception as error:
                st.error(
                    "Image processing failed. "
                    "Check the model files and application logs."
                )
                st.exception(error)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "whatsproof_result" in st.session_state:

    result = st.session_state["whatsproof_result"]

    st.divider()
    st.header("Degradation Estimation Results")

    # Classification.
    predicted_class = result["predicted_class"]

    if predicted_class == "Forwarded":
        st.warning("Prediction: Forwarded image")
    else:
        st.success("Prediction: Original image")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Class",
            predicted_class,
        )

        st.metric(
            "Predicted-Class Probability",
            f"{result['model_probability']:.1%}",
        )

    with col2:
        st.metric(
            "Degradation Indicator",
            f"{result['degradation_score']:.1f}/100",
        )

        st.metric(
            "Estimated Degradation Level",
            result["degradation_level"],
        )

    st.write("**Interpretation:**", result["explanation"])

    st.caption(
        result.get(
            "degradation_score_note",
            "The degradation indicator is a dataset-relative heuristic.",
        )
    )

    # Optional feature-level indicators.
    component_scores = result.get("component_scores", {})

    if component_scores:
        with st.expander("View feature-level indicators"):
            for feature_name, score in component_scores.items():
                st.write(f"**{feature_name}:** {score:.1f}/100")

    # --------------------------------------------------------
    # BEFORE / AFTER COMPARISON
    # --------------------------------------------------------

    st.divider()
    st.header("Image Enhancement Comparison")

    original_col, enhanced_col = st.columns(2)

    original_image_bytes = st.session_state[
        "whatsproof_original_bytes"
    ]

    enhanced_image_bytes = st.session_state[
        "whatsproof_enhanced_bytes"
    ]

    with original_col:
        st.subheader("Before Enhancement")

        st.image(
            original_image_bytes,
            width="stretch",
        )

        st.download_button(
            label="Download Original Image",
            data=original_image_bytes,
            file_name=st.session_state["whatsproof_filename"],
            mime="application/octet-stream",
            use_container_width=True,
        )

    with enhanced_col:
        st.subheader("After Enhancement")

        st.image(
            enhanced_image_bytes,
            width="stretch",
        )

        enhanced_filename = (
            f"{Path(st.session_state['whatsproof_filename']).stem}"
            "_enhanced.png"
        )

        st.download_button(
            label="Download Enhanced Image",
            data=enhanced_image_bytes,
            file_name=enhanced_filename,
            mime="image/png",
            use_container_width=True,
        )

    st.caption(
        "Enhancement applies conservative denoising, luminance/contrast "
        "adjustment, and sharpening. It does not guarantee improved "
        "perceived quality or a more accurate classification."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "WhatsProof | Final Year Project | "
    "Image degradation estimation and enhancement"
)
# code
