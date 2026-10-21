import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

API_USER = os.getenv("SIGHTENGINE_API_USER")
API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")

API_URL = "https://api.sightengine.com/1.0/check.json"


def detect_ai_image(image_path):
    """Send one image to Sightengine and return its AI-generated score."""

    image_path = Path(image_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not API_USER or not API_SECRET:
        raise RuntimeError(
            "Sightengine credentials missing. Check the project-root .env file."
        )

    print(f"\nAnalyzing: {image_path.name}")

    params = {
        "models": "genai",
        "api_user": API_USER,
        "api_secret": API_SECRET,
    }

    with image_path.open("rb") as image_file:
        response = requests.post(
            API_URL,
            data=params,
            files={
                "media": (
                    image_path.name,
                    image_file,
                    "application/octet-stream",
                )
            },
            timeout=90,
        )

    response.raise_for_status()
    result = response.json()

    if result.get("status") != "success":
        raise RuntimeError(
            f"API error for {image_path.name}: "
            f"{result.get('error', result)}"
        )

    score = result.get("type", {}).get("ai_generated")

    if score is None:
        raise RuntimeError(
            f"No AI-generated score returned for {image_path.name}: {result}"
        )

    score = float(score)

    if not 0 <= score <= 1:
        raise ValueError(f"Unexpected AI score: {score}")

    return score * 100


def compare_original_and_enhanced(original_path, enhanced_path):
    """Analyze both images and display their scores side by side."""

    original_path = Path(original_path)
    enhanced_path = Path(enhanced_path)

    # Resolve paths relative to the current terminal directory.
    if not original_path.is_absolute():
        original_path = Path.cwd() / original_path

    if not enhanced_path.is_absolute():
        enhanced_path = Path.cwd() / enhanced_path

    original_path = original_path.resolve()
    enhanced_path = enhanced_path.resolve()

    if not original_path.is_file():
        raise FileNotFoundError(f"Original image not found: {original_path}")

    if not enhanced_path.is_file():
        raise FileNotFoundError(f"Enhanced image not found: {enhanced_path}")

    print("\n========================================")
    print("       WhatsProof AI Detection")
    print("========================================")

    # Both images are sent independently to the API.
    original_score = detect_ai_image(original_path)
    enhanced_score = detect_ai_image(enhanced_path)

    original_label = (
        "LIKELY AI-GENERATED"
        if original_score >= 50
        else "LIKELY NOT AI-GENERATED"
    )

    enhanced_label = (
        "LIKELY AI-GENERATED"
        if enhanced_score >= 50
        else "LIKELY NOT AI-GENERATED"
    )

    score_change = enhanced_score - original_score

    print("\n============== RESULTS ==============")
    print(f"Original image : {original_label}")
    print(f"AI score before enhancement: {original_score:.2f}%")
    print("-------------------------------------")
    print(f"Enhanced image : {enhanced_label}")
    print(f"AI score after enhancement : {enhanced_score:.2f}%")
    print("-------------------------------------")
    print(f"Score change (after - before): {score_change:+.2f} percentage points")
    print("=====================================")

    # Display both images and their AI scores.
    original_img = mpimg.imread(str(original_path))
    enhanced_img = mpimg.imread(str(enhanced_path))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(original_img)
    axes[0].set_title(
        f"BEFORE ENHANCEMENT\n"
        f"AI-generated score: {original_score:.2f}%\n"
        f"{original_label}",
        fontsize=12,
    )
    axes[0].axis("off")

    axes[1].imshow(enhanced_img)
    axes[1].set_title(
        f"AFTER ENHANCEMENT\n"
        f"AI-generated score: {enhanced_score:.2f}%\n"
        f"{enhanced_label}",
        fontsize=12,
    )
    axes[1].axis("off")

    fig.suptitle("WhatsProof: Before vs After Enhancement", fontsize=16)
    plt.tight_layout()

    # Save comparison output.
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / "ai_detection_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")

    print(f"\nComparison image saved to:\n{comparison_path}")

    plt.show()

    return {
        "original_ai_score": original_score,
        "enhanced_ai_score": enhanced_score,
        "score_change": score_change,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            'Usage: python ai_detector_api.py '
            '"path_to_original_image" "path_to_enhanced_image"'
        )
        sys.exit(1)

    try:
        compare_original_and_enhanced(sys.argv[1], sys.argv[2])
    except Exception as error:
        print(f"\nDetection failed: {error}")
        sys.exit(1)
