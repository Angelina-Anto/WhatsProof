# WhatsProof — WhatsApp Image Degradation Analysis and Enhancement

WhatsProof is a final-year project that investigates how forwarding images through WhatsApp affects image quality and how image processing techniques can help improve the visual appearance of forwarded images.

The project combines a machine-learning-based degradation estimator with conservative image enhancement in a Streamlit web interface. Users can upload an image, analyze its characteristics, view the estimated degradation level, and compare the original image with the enhanced result.

## Features

- **Image upload:** Upload an image through the Streamlit web interface.
- **Degradation estimation:** Classify an image as Original or Forwarded using a trained machine learning model.
- **Degradation indicator:** Display a dataset-relative indicator on a scale of 0–100, alongside a descriptive degradation category.
- **Image enhancement:** Apply conservative denoising, local contrast enhancement, and sharpening.
- **Before-and-after comparison:** View the uploaded image and enhanced image side by side.
- **Web interface:** Access the workflow through a Streamlit application without running individual scripts manually.

## Project Workflow

The current implementation follows this workflow:

1. **Upload an image** using the Streamlit interface.
2. **Extract image features**, including image statistics and compression-related characteristics.
3. **Estimate degradation** using the previously trained classification model.
4. **Enhance the image** using conservative image-processing operations.
5. **Display the results**, including the predicted class, degradation indicator, descriptive category, and original-versus-enhanced comparison.

## Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pandas
- Pillow
- scikit-learn
- Joblib
- Matplotlib

## Repository Structure

```text
WhatsProof/
├── app.py
├── test_degradation_estimator.py
├── image_enhancement.py
├── degradation_estimator_model.joblib
├── degradation_estimator_features.json
├── requirements.txt
└── README.md
```

### File descriptions

| File | Purpose |
|---|---|
| `app.py` | Runs the Streamlit web interface. |
| `test_degradation_estimator.py` | Loads the trained model, extracts features from an uploaded image, and estimates its class and degradation indicator. |
| `image_enhancement.py` | Applies conservative image enhancement. |
| `degradation_estimator_model.joblib` | Stores the trained machine learning model and its associated prediction information. |
| `degradation_estimator_features.json` | Stores feature schema and model metadata. |
| `requirements.txt` | Lists the Python dependencies required to run the application. |

The training script and dataset are not required to run the deployed interface because the trained model is already included in the repository.

## Running the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/Angelina-Anto/WhatsProof
cd WhatsProof
```


### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Compatibility note:** Use the same scikit-learn version that was used to train and save the model. The version should be pinned in `requirements.txt` to avoid model-loading compatibility issues.

### 4. Launch Streamlit

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`, where you can access the application.

## Deploying on Streamlit Community Cloud

1. Push the project files to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select the WhatsProof repository and its branch.
5. Set the main application file to `app.py`.
6. Deploy the application.

Ensure that the trained model, feature schema, and `requirements.txt` are present in the repository at the paths expected by the code.

## Degradation Estimation

The degradation estimator uses image statistics and compression-related features to classify images into two categories:

- **Original**
- **Forwarded**

The estimator also provides a degradation indicator and a corresponding descriptive category, such as Low, Mild, Moderate, High, or Very High Degradation.

The indicator is a **dataset-relative heuristic**, not a scientifically calibrated percentage of image quality lost. Its values depend on the training dataset and the features learned from that dataset.

## Image Enhancement

The enhancement stage applies the following operations:

1. **Mild noise reduction** to reduce some visible noise while attempting to preserve edges.
2. **Contrast enhancement** using CLAHE in the LAB color space to improve local luminance contrast.
3. **Conservative sharpening** to make edges and details appear clearer.

The purpose is to improve visual appearance without unnecessarily altering the input image. Enhancement does not guarantee recovery of information lost during forwarding.

## Dataset and Privacy

The original training dataset is not included in this repository.

For deployment, the application processes the image supplied by the user. Images should be handled responsibly, particularly if they contain personal or sensitive information. Avoid uploading private images to public repositories, and review the application code and hosting configuration to understand how uploaded images and generated outputs are handled.

Do not commit `.env` files, API credentials, private datasets, or sensitive user images to GitHub.

## Current Scope

The current implementation focuses on:

- Binary Original-versus-Forwarded classification.
- Dataset-relative degradation estimation.
- Conservative image enhancement.
- Interactive before-and-after image comparison.

AI-generated image detection is not part of the current deployed workflow.

## Limitations

- Classification performance depends on the dataset, labels, and collection procedure.
- Forwarding-related changes may overlap with ordinary image variations.
- The degradation indicator is heuristic and should not be interpreted as a precise physical measurement.
- Enhancement may improve appearance but cannot reliably reconstruct all information lost through compression.
- Results on the training dataset do not guarantee equivalent performance on images from other devices, platforms, or forwarding workflows.

## Future Work

Potential extensions include:

- Integrating AI-generated image detection into the application.
- Evaluating the estimator on larger and more diverse datasets.
- Improving validation across different devices and forwarding conditions.
- Comparing additional enhancement techniques.
- Conducting systematic before-and-after evaluations using image-quality metrics.

## Project Information

**Project:** WhatsProof  
**Project type:** Final-year academic project  
**Focus:** Image degradation estimation and image enhancement for WhatsApp-forwarded images

---

*WhatsProof is an academic research prototype. Its outputs are estimates intended for analysis and demonstration, not definitive proof of an image's origin or history.*
