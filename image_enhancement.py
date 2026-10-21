
from pathlib import Path
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


def enhance_image(image_path, output_path):
    """Enhance an image conservatively and save a separate copy."""

    image_path = Path(image_path)
    output_path = Path(output_path)

    # Load image without modifying the original
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 1. Reduce mild noise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 2. Apply mild luminance/contrast correction in LAB color space
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )
    adjusted_luminance = clahe.apply(luminance)

    adjusted_lab = cv2.merge(
        [adjusted_luminance, channel_a, channel_b]
    )
    contrast_corrected = cv2.cvtColor(
        adjusted_lab, cv2.COLOR_LAB2BGR
    )

    # 3. Apply conservative sharpening
    blurred = cv2.GaussianBlur(
        contrast_corrected, (0, 0), 1.0
    )

    enhanced = cv2.addWeighted(
        contrast_corrected, 1.15,
        blurred, -0.15,
        0
    )

    # Save as a separate image
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), enhanced)
    if not success:
        raise OSError(f"Could not save image: {output_path}")

    print(f"Original:  {image_path}")
    print(f"Enhanced:  {output_path}")
    print(f"Dimensions: {image.shape[1]} x {image.shape[0]}")

    # Display the comparison
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

    return enhanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WhatsProof image enhancement"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--output",
        default="outputs/enhanced_image.png",
        help="Path to save the enhanced image"
    )

    args = parser.parse_args()
    enhance_image(args.input, args.output)
