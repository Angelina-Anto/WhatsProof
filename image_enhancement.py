
from pathlib import Path
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


def enhance_image(image_path, output_path, show_comparison=False):
    """Enhance an image conservatively and save a separate copy.

    Args:
        image_path: Path to the original image.
        output_path: Path where the enhanced image will be saved.
        show_comparison: Show a Matplotlib comparison when True.

    Returns:
        The enhanced image as an OpenCV BGR NumPy array.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the original without modifying it.
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 1. Reduce mild noise while preserving edges.
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # 2. Apply mild luminance/contrast correction in LAB space.
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    )

    adjusted_luminance = clahe.apply(luminance)

    adjusted_lab = cv2.merge(
        [adjusted_luminance, channel_a, channel_b]
    )

    contrast_corrected = cv2.cvtColor(
        adjusted_lab,
        cv2.COLOR_LAB2BGR,
    )

    # 3. Apply conservative sharpening.
    blurred = cv2.GaussianBlur(
        contrast_corrected,
        (0, 0),
        1.0,
    )

    enhanced = cv2.addWeighted(
        contrast_corrected,
        1.15,
        blurred,
        -0.15,
        0,
    )

    # Save separately; never overwrite the original.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), enhanced)

    if not success:
        raise OSError(f"Could not save enhanced image: {output_path}")

    print(f"Original:   {image_path}")
    print(f"Enhanced:   {output_path}")
    print(f"Dimensions: {image.shape[1]} x {image.shape[0]}")

    # Optional local comparison. Streamlit displays images itself.
    if show_comparison:
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(enhanced_rgb)
        axes[1].set_title("Enhanced Image")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return enhanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WhatsProof image enhancement"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image",
    )

    parser.add_argument(
        "--output",
        default="outputs/enhanced_image.png",
        help="Path to save the enhanced image",
    )

    args = parser.parse_args()

    enhance_image(
        args.input,
        args.output,
        show_comparison=True,
    )
