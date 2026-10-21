
from pathlib import Path
import joblib

# Locate the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Search for saved joblib models in the project
model_files = list(PROJECT_DIR.rglob("*.joblib"))

if not model_files:
    print("No .joblib model found in the project.")
else:
    print("Found model files:")
    for i, path in enumerate(model_files, start=1):
        print(f"{i}. {path}")

    # Load the first model for a basic loading test
    model_path = model_files[0]
    model = joblib.load(model_path)

    print("\nModel loaded successfully!")
    print("Model path:", model_path)
    print("Model type:", type(model).__name__)

    if hasattr(model, "n_features_in_"):
        print("Expected feature count:", model.n_features_in_)

    if hasattr(model, "classes_"):
        print("Model classes:", model.classes_)
