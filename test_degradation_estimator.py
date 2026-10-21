
"""
WhatsProof - Degradation Estimator Prediction Module

Purpose:
    Predict whether an image is Original or Forwarded using the
    previously trained model.

    Also calculate the dataset-relative degradation indicator,
    explain the result in plain language, and optionally save an
    annotated preview.

Usage from project root:
    python src/test_degradation_estimator.py --predict "path/to/image.jpg"

Usage from app.py:
    from src.test_degradation_estimator import predict_image
    result = predict_image(image_path)

Required files in project root:
    degradation_estimator_model.joblib
    degradation_estimator_features.json

No dataset or training report is required for prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"

MODEL_FILE = ROOT / "degradation_estimator_model.joblib"
SCHEMA_FILE = ROOT / "degradation_estimator_features.json"


# ============================================================
# CLASS NAMES
# ============================================================

NAMES = {
    0: "Original",
    1: "Forwarded",
}


# ============================================================
# PLAIN-LANGUAGE DEGRADATION EXPLANATION
# Keep thresholds consistent with the training script.
# ============================================================

def explain_degradation(score: float) -> tuple[str, str]:
    """Return a readable degradation category and explanation."""

    if score < 20:
        return (
            "LOW DEGRADATION",
            "The image shows few signs of forwarding-related changes.",
        )

    if score < 40:
        return (
            "MILD DEGRADATION",
            "The image shows some signs of quality changes.",
        )

    if score < 60:
        return (
            "MODERATE DEGRADATION",
            "The image shows noticeable signs of quality changes.",
        )

    if score < 80:
        return (
            "HIGH DEGRADATION",
            "The image shows strong signs of quality changes.",
        )

    return (
        "VERY HIGH DEGRADATION",
        "The image shows very strong signs of quality changes.",
    )


# ============================================================
# IMAGE STATISTICS
# These functions match the training feature calculations.
# ============================================================

def entropy(gray: np.ndarray) -> float:
    frequencies = np.bincount(
        gray.ravel(),
        minlength=256,
    ).astype(np.float64)

    probabilities = frequencies[frequencies > 0]
    probabilities /= max(probabilities.sum(), 1)

    return float(
        -(probabilities * np.log2(probabilities)).sum()
    )


def block_metrics(gray: np.ndarray) -> dict[str, float]:
    """Estimate differences at 8-pixel block boundaries."""

    image = gray.astype(np.float32)

    horizontal_diff = np.abs(np.diff(image, axis=1))
    vertical_diff = np.abs(np.diff(image, axis=0))

    horizontal_boundaries = (
        np.arange(1, image.shape[1]) % 8
    ) == 0

    vertical_boundaries = (
        np.arange(1, image.shape[0]) % 8
    ) == 0

    horizontal_edge = (
        float(horizontal_diff[:, horizontal_boundaries].mean())
        if horizontal_boundaries.any()
        else 0.0
    )

    horizontal_other = (
        float(horizontal_diff[:, ~horizontal_boundaries].mean())
        if (~horizontal_boundaries).any()
        else 0.0
    )

    vertical_edge = (
        float(vertical_diff[vertical_boundaries, :].mean())
        if vertical_boundaries.any()
        else 0.0
    )

    vertical_other = (
        float(vertical_diff[~vertical_boundaries, :].mean())
        if (~vertical_boundaries).any()
        else 0.0
    )

    return {
        "block_h_edge": horizontal_edge,
        "block_h_nonedge": horizontal_other,
        "block_v_edge": vertical_edge,
        "block_v_nonedge": vertical_other,
        "block_h_excess": horizontal_edge - horizontal_other,
        "block_v_excess": vertical_edge - vertical_other,
    }


def dct_metrics(gray: np.ndarray) -> dict[str, float]:
    """Calculate frequency-domain features using 8x8 DCT blocks."""

    height = (gray.shape[0] // 8) * 8
    width = (gray.shape[1] // 8) * 8

    if height < 8 or width < 8:
        return {
            "dct_mid_energy": 0.0,
            "dct_high_energy": 0.0,
            "dct_high_fraction": 0.0,
            "dct_zero_ac_fraction": 0.0,
        }

    image = gray[:height, :width].astype(np.float32) - 128.0

    blocks = (
        image.reshape(height // 8, 8, width // 8, 8)
        .transpose(0, 2, 1, 3)
    )

    n = np.arange(8, dtype=np.float32)
    k = np.arange(8, dtype=np.float32)[:, None]

    transform = np.cos(
        np.pi * (2 * n[None, :] + 1) * k / 16.0
    ).astype(np.float32)

    transform[0] *= np.sqrt(1.0 / 8.0)
    transform[1:] *= np.sqrt(2.0 / 8.0)

    coefficients = np.einsum(
        "ui,...ij,vj->...uv",
        transform,
        blocks,
        transform,
        optimize=True,
    )

    energy = coefficients * coefficients

    frequencies = (
        np.arange(8)[:, None]
        + np.arange(8)[None, :]
    )

    mid = energy[..., (frequencies >= 3) & (frequencies <= 7)]
    high = energy[..., frequencies >= 8]
    ac = energy[..., frequencies > 0]

    return {
        "dct_mid_energy": float(mid.mean()),
        "dct_high_energy": float(high.mean()),
        "dct_high_fraction": float(
            high.sum() / (ac.sum() + 1e-8)
        ),
        "dct_zero_ac_fraction": float(
            (
                np.abs(coefficients[..., frequencies > 0]) < 0.5
            ).mean()
        ),
    }


# ============================================================
# FEATURE EXTRACTION
# IMPORTANT: Keep consistent with the training script.
# ============================================================

def extract_features(path: Path) -> dict[str, float]:
    """Extract the same image features used during training."""

    path = Path(path)

    with Image.open(path) as image:
        image_format = (image.format or "UNKNOWN").upper()

        width, height = image.size
        rgb = image.convert("RGB")

        small_rgb = rgb.copy()
        small_rgb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        gray_image = small_rgb.convert("L")
        gray = np.asarray(gray_image, dtype=np.uint8)

        rgb_stats = ImageStat.Stat(small_rgb)
        gray_stats = ImageStat.Stat(gray_image)

        file_size = path.stat().st_size

        features = {
            "width": float(width),
            "height": float(height),
            "aspect_ratio": width / max(height, 1),
            "file_size": float(file_size),
            "bytes_per_pixel": file_size / max(width * height, 1),

            "is_jpeg": float(image_format in ("JPEG", "JPG")),
            "is_png": float(image_format == "PNG"),
            "is_webp": float(image_format == "WEBP"),

            "mean_r": float(rgb_stats.mean[0]),
            "mean_g": float(rgb_stats.mean[1]),
            "mean_b": float(rgb_stats.mean[2]),

            "std_r": float(rgb_stats.stddev[0]),
            "std_g": float(rgb_stats.stddev[1]),
            "std_b": float(rgb_stats.stddev[2]),

            "gray_mean": float(gray_stats.mean[0]),
            "gray_std": float(gray_stats.stddev[0]),
            "entropy": entropy(gray),
        }

        edge_image = gray_image.filter(ImageFilter.FIND_EDGES)
        edge_stats = ImageStat.Stat(edge_image)

        features["edge_mean"] = float(edge_stats.mean[0])
        features["edge_std"] = float(edge_stats.stddev[0])

        # Laplacian variance.
        array = gray.astype(np.float32)

        laplacian = (
            -4 * array[1:-1, 1:-1]
            + array[1:-1, :-2]
            + array[1:-1, 2:]
            + array[:-2, 1:-1]
            + array[2:, 1:-1]
        )

        features["laplacian_variance"] = (
            float(np.var(laplacian))
            if laplacian.size
            else 0.0
        )

        # Gradient statistics.
        gradient_y, gradient_x = np.gradient(array)
        gradient = np.hypot(gradient_x, gradient_y)

        features["gradient_mean"] = float(gradient.mean())
        features["gradient_std"] = float(gradient.std())

        # Median-filter residual statistics.
        median_image = np.asarray(
            gray_image.filter(ImageFilter.MedianFilter(size=3)),
            dtype=np.float32,
        )

        features["median_residual_mean"] = float(
            np.abs(array - median_image).mean()
        )

        features["median_residual_std"] = float(
            np.std(array - median_image)
        )

        # JPEG quantization-table features, when available.
        tables = getattr(image, "quantization", None)

        quantization_values = (
            [
                float(value)
                for table in tables.values()
                for value in table
            ]
            if tables
            else []
        )

        features.update({
            "has_jpeg_quant_table": float(bool(quantization_values)),
            "jpeg_quant_mean": (
                float(np.mean(quantization_values))
                if quantization_values
                else 0.0
            ),
            "jpeg_quant_std": (
                float(np.std(quantization_values))
                if quantization_values
                else 0.0
            ),
            "jpeg_quant_min": (
                float(np.min(quantization_values))
                if quantization_values
                else 0.0
            ),
            "jpeg_quant_max": (
                float(np.max(quantization_values))
                if quantization_values
                else 0.0
            ),
        })

        compression_sample = np.asarray(
            rgb.crop((
                0,
                0,
                min(width, 512),
                min(height, 512),
            )).convert("L")
        )

        features.update(block_metrics(compression_sample))
        features.update(dct_metrics(compression_sample))

        return features


# ============================================================
# DEGRADATION INDICATOR
# Uses the baseline saved inside the trained model bundle.
# ============================================================

def calculate_degradation_score(
    features: dict,
    baseline: dict,
) -> tuple[float, dict[str, float]]:
    """
    Estimate movement from Original training medians toward
    Forwarded training medians.

    This is a dataset-relative heuristic, not literal quality loss.
    """

    component_scores = {}

    for name, stats in baseline.get("features", {}).items():
        if name not in features:
            continue

        value = float(features[name])
        original = float(stats["original_median"])
        forwarded = float(stats["forwarded_median"])
        direction = float(stats["direction"])

        difference = abs(forwarded - original)

        if difference < 1e-8:
            continue

        progress = (
            (value - original) * direction
        ) / difference

        component_scores[name] = float(
            np.clip(progress, 0.0, 1.0) * 100.0
        )

    if not component_scores:
        return 0.0, {}

    return (
        float(np.mean(list(component_scores.values()))),
        component_scores,
    )


# ============================================================
# ANNOTATED IMAGE
# ============================================================

def save_annotated_analysis(
    image_path: Path,
    predicted_class: str,
    model_probability: float,
    degradation_score: float,
    output_path: Path,
) -> None:
    """Save an annotated preview without changing the input image."""

    level, explanation = explain_degradation(degradation_score)

    with Image.open(image_path) as source:
        preview = source.convert("RGB")
        preview.thumbnail((1200, 900), Image.Resampling.LANCZOS)

        panel_height = 160

        canvas = Image.new(
            "RGB",
            (preview.width, preview.height + panel_height),
            "white",
        )

        canvas.paste(preview, (0, panel_height))
        draw = ImageDraw.Draw(canvas)

        try:
            title_font = ImageFont.truetype("arial.ttf", 23)
            text_font = ImageFont.truetype("arial.ttf", 17)
        except OSError:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()

        draw.text(
            (15, 10),
            f"WhatsProof: {predicted_class}",
            fill="black",
            font=title_font,
        )

        draw.text(
            (15, 45),
            f"Degradation level: {level}",
            fill="black",
            font=text_font,
        )

        draw.text(
            (15, 75),
            explanation,
            fill="black",
            font=text_font,
        )

        draw.text(
            (15, 105),
            f"Degradation indicator: {degradation_score:.1f}/100",
            fill="black",
            font=text_font,
        )

        draw.text(
            (15, 132),
            f"Predicted-class probability: {model_probability:.1%}",
            fill="black",
            font=text_font,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)


# ============================================================
# LOAD TRAINED MODEL AND FEATURE SCHEMA
# ============================================================

def load_model_bundle() -> dict:
    """Load the existing model and feature schema."""

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Ensure degradation_estimator_model.joblib is in the "
            "project root."
        )

    if not SCHEMA_FILE.is_file():
        raise FileNotFoundError(
            f"Feature schema not found: {SCHEMA_FILE}\n"
            "Ensure degradation_estimator_features.json is in the "
            "project root."
        )

    # Only load trusted model files. Joblib/pickle files can execute
    # code when loaded if they have been maliciously constructed.
    bundle = joblib.load(MODEL_FILE)

    with SCHEMA_FILE.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(
            "Unexpected model file format. Expected a dictionary "
            "containing the trained model under the 'model' key."
        )

    # Prefer the exact schema stored in the trained bundle.
    # Fall back to the separately saved JSON where appropriate.
    feature_columns = bundle.get("feature_columns")
    if not feature_columns:
        feature_columns = schema.get("feature_columns")

    if not feature_columns:
        raise ValueError(
            "No feature_columns found in model bundle or schema JSON."
        )

    bundle["feature_columns"] = feature_columns

    if "degradation_baseline" not in bundle:
        bundle["degradation_baseline"] = schema.get(
            "degradation_baseline",
            {"features": {}},
        )

    return bundle


# ============================================================
# PUBLIC PREDICTION FUNCTION FOR STREAMLIT
# ============================================================

def predict_image(
    image_path: str | Path,
    save_preview: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Analyze one image and return structured results for app.py.

    Parameters
    ----------
    image_path:
        Path to the image to analyze.
    save_preview:
        Whether to save the annotated analysis image.
    output_dir:
        Optional directory for the annotated preview. Defaults to
        the project's outputs directory.

    Returns
    -------
    dict:
        predicted_class
        predicted_label
        model_probability
        degradation_score
        degradation_level
        explanation
        component_scores
        features
        preview_path
    """

    image_path = Path(image_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Validate that the image can be opened before feature extraction.
    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(
            f"The uploaded file is not a valid supported image: "
            f"{image_path.name}"
        ) from exc

    bundle = load_model_bundle()
    feature_columns = bundle["feature_columns"]

    features = extract_features(image_path)

    # Reindex to exactly the columns and order used during training.
    one_image = pd.DataFrame([features]).reindex(
        columns=feature_columns
    )

    # Predict Original (0) or Forwarded (1).
    label = int(bundle["model"].predict(one_image)[0])
    predicted_class = NAMES.get(label, str(label))

    # Probability for the predicted class, consistent with train script.
    probability = 0.0

    if hasattr(bundle["model"], "predict_proba"):
        probabilities = bundle["model"].predict_proba(one_image)[0]

        probability_by_label = dict(
            zip(bundle["model"].classes_, probabilities)
        )

        probability = float(probability_by_label[label])

    baseline = bundle.get(
        "degradation_baseline",
        {"features": {}},
    )

    degradation_score, component_scores = calculate_degradation_score(
        features,
        baseline,
    )

    degradation_level, explanation = explain_degradation(
        degradation_score
    )

    preview_path = None

    if save_preview:
        destination = (
            Path(output_dir).resolve()
            if output_dir is not None
            else OUTPUTS_DIR
        )

        destination.mkdir(parents=True, exist_ok=True)

        preview_path = destination / (
            f"{image_path.stem}_degradation_analysis.png"
        )

        save_annotated_analysis(
            image_path=image_path,
            predicted_class=predicted_class,
            model_probability=probability,
            degradation_score=degradation_score,
            output_path=preview_path,
        )

    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "predicted_label": label,
        "predicted_class": predicted_class,
        "model_probability": probability,
        "degradation_score": degradation_score,
        "degradation_level": degradation_level,
        "explanation": explanation,
        "component_scores": component_scores,
        "features": features,
        "preview_path": (
            str(preview_path) if preview_path is not None else None
        ),
        "degradation_score_note": (
            "Dataset-relative heuristic indicator. It is not a "
            "literal percentage of pixels or quality lost."
        ),
    }


# ============================================================
# COMMAND-LINE OUTPUT
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WhatsProof: Analyze an image using the trained estimator."
    )

    parser.add_argument(
        "--predict",
        type=Path,
        required=True,
        help="Path to an image to analyze.",
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not save an annotated preview image.",
    )

    args = parser.parse_args()

    try:
        result = predict_image(
            args.predict,
            save_preview=not args.no_preview,
        )

        print("\n========== WHATSProof IMAGE ANALYSIS ==========")
        print(f"Image: {result['image_name']}")
        print(f"Classification: {result['predicted_class']}")
        print(f"Degradation level: {result['degradation_level']}")
        print(f"What this means: {result['explanation']}")
        print(
            f"Degradation indicator: "
            f"{result['degradation_score']:.1f}/100"
        )
        print(
            f"Predicted-class probability: "
            f"{result['model_probability']:.1%}"
        )

        print(f"\nNote: {result['degradation_score_note']}")

        if result["component_scores"]:
            print("\nFeature-level indicators:")

            for feature_name, score in result["component_scores"].items():
                print(f"  {feature_name}: {score:.1f}/100")

        if result["preview_path"]:
            print(
                f"\nAnnotated analysis saved to:\n"
                f"{result['preview_path']}"
            )

    except Exception as error:
        parser.exit(1, f"\nPrediction failed: {error}\n")


if __name__ == "__main__":
    main()
