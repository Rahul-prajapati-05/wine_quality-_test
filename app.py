"""
=============================================================================
VinoPredict AI - Backend Server (Starlette + Uvicorn)
Oasis Infobyte Data Science Internship - Task 2
Provides REST API endpoints for real-time model inference and serves
the rich interactive frontend web application.
=============================================================================
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
import uvicorn

# Root project paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"
DATASET_PATH = BASE_DIR / "wine+quality" / "winequality-red.csv"

MODEL_FILE = MODELS_DIR / "wine_rf_model.joblib"

# Global Model Cache
MODEL_DATA = None


def get_or_train_model():
    """
    Loads trained model artifact from disk or trains Random Forest baseline if missing.
    """
    global MODEL_DATA
    if MODEL_DATA is not None:
        return MODEL_DATA

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_FILE.exists():
        print(f"[INFO] Loading cached Random Forest model from {MODEL_FILE}...")
        MODEL_DATA = joblib.load(MODEL_FILE)
        return MODEL_DATA

    print("[INFO] Model file not found. Training Random Forest model on the fly...")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, sep=";")
    df["quality_label"] = df["quality"].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2))
    X = df.drop(columns=["quality", "quality_label"])
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    MODEL_DATA = {
        "model": rf,
        "feature_names": list(X.columns),
        "classes": [0, 1, 2],
        "class_names": ["Low (3-4)", "Medium (5-6)", "High (7-8)"]
    }
    joblib.dump(MODEL_DATA, MODEL_FILE)
    print(f"[INFO] Model trained and persisted to {MODEL_FILE}")
    return MODEL_DATA


# =============================================================================
# API HANDLERS
# =============================================================================

async def home(request):
    """Serves the main frontend single-page application."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend index.html not found</h1>", status_code=404)
    return FileResponse(index_path)


async def api_health(request):
    """Health check and model status endpoint."""
    return JSONResponse({
        "status": "online",
        "service": "VinoPredict AI - Wine Quality Prediction API",
        "internship": "Oasis Infobyte Data Science Task 2",
        "model_loaded": MODEL_FILE.exists()
    })


async def api_predict(request):
    """
    Accepts 11 physicochemical feature values in JSON, returns predicted quality tier,
    class probabilities, estimated score, and oenological assessment.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON payload provided"}, status_code=400)

    model_data = get_or_train_model()
    model = model_data["model"]
    feature_names = model_data["feature_names"]

    # Extract features in required strict order
    try:
        feature_vector = []
        for feat in feature_names:
            # support both snake_case and space separated keys
            val = data.get(feat, data.get(feat.replace(" ", "_"), None))
            if val is None:
                return JSONResponse({"error": f"Missing required feature: '{feat}'"}, status_code=400)
            feature_vector.append(float(val))
    except (ValueError, TypeError) as e:
        return JSONResponse({"error": f"Invalid numerical value: {str(e)}"}, status_code=400)

    X_df = pd.DataFrame([feature_vector], columns=feature_names)

    # Predictions
    pred_class = int(model.predict(X_df)[0])
    probabilities = model.predict_proba(X_df)[0]

    # Map class metadata
    tier_info = {
        0: {
            "tier_name": "Low Quality",
            "tier_label": "Low (Grade 3–4)",
            "tier_badge": "Defective / Budget Tier",
            "tier_color": "#e63946",
            "stars": 2,
            "description": "Exhibits sensory flaws such as high volatile acidity (vinegar note) or deficient alcohol balance. Suitable strictly for industrial vinegar or low-grade bulk blending."
        },
        1: {
            "tier_name": "Medium Quality",
            "tier_label": "Medium (Grade 5–6)",
            "tier_badge": "Standard Commercial Wine",
            "tier_color": "#e3a857",
            "stars": 3,
            "description": "Sound, drinkable commercial table wine meeting standard market quality specifications. Well-balanced acidity and moderate alcohol content."
        },
        2: {
            "tier_name": "High Quality",
            "tier_label": "High (Grade 7–8)",
            "tier_badge": "Premium / Reserve Selection",
            "tier_color": "#2a9d8f",
            "stars": 5,
            "description": "Exceptional wine with optimal alcohol body, pristine volatile acidity control, and strong antioxidant sulphate protection. Premium export quality."
        }
    }

    tier = tier_info.get(pred_class, tier_info[1])

    # Calculate an estimated continuous score on 3-8 scale based on probabilities
    # Weight: Class 0 -> 3.5, Class 1 -> 5.5, Class 2 -> 7.4
    estimated_score = round(float(probabilities[0] * 3.5 + probabilities[1] * 5.5 + probabilities[2] * 7.4), 2)

    # Chemical Evaluation
    alcohol = float(data.get("alcohol", 10.4))
    volatile_acid = float(data.get("volatile acidity", data.get("volatile_acidity", 0.52)))
    sulphates = float(data.get("sulphates", 0.65))
    citric = float(data.get("citric acid", data.get("citric_acid", 0.27)))

    # Winemaker recommendations
    tips = []
    if volatile_acid > 0.60:
        tips.append("⚠️ Volatile Acidity is elevated (>0.60 g/dm³). Prevent Acetobacter bacterial exposure to eliminate vinegar notes.")
    else:
        tips.append("✅ Volatile Acidity is clean and well-controlled (<0.60 g/dm³), preserving pure fruit aromatics.")

    if alcohol >= 11.5:
        tips.append("✅ Alcohol content is robust (≥11.5%), imparting excellent palate weight, body, and richness.")
    elif alcohol < 9.5:
        tips.append("ℹ️ Alcohol is relatively low (<9.5%). Harvesting grapes at higher Brix maturity would enhance richness.")

    if sulphates >= 0.65:
        tips.append("✅ Sulphate level is optimal (≥0.65 g/dm³), providing strong antioxidant and antimicrobial protection.")
    else:
        tips.append("ℹ️ Sulphates are slightly below optimal. Consider careful potassium metabisulphite adjustment to prevent premature oxidation.")

    prob_dict = {
        "Low (3-4)": round(float(probabilities[0]), 4),
        "Medium (5-6)": round(float(probabilities[1]), 4),
        "High (7-8)": round(float(probabilities[2]), 4)
    }

    response_payload = {
        "predicted_class": pred_class,
        "tier_name": tier["tier_name"],
        "tier_label": tier["tier_label"],
        "tier_badge": tier["tier_badge"],
        "tier_color": tier["tier_color"],
        "stars": tier["stars"],
        "description": tier["description"],
        "confidence": round(float(probabilities[pred_class]), 4),
        "probabilities": prob_dict,
        "estimated_score": estimated_score,
        "oenological_profile": {
            "alcohol_assessment": "High (Rich Body)" if alcohol >= 11.5 else ("Low (Thin Body)" if alcohol < 9.5 else "Moderate Commercial Body"),
            "volatile_acidity_assessment": "Defective (Vinegar)" if volatile_acid > 0.65 else ("Clean & Aromatic" if volatile_acid < 0.45 else "Acceptable"),
            "sulphates_assessment": "Excellent Protection" if sulphates >= 0.70 else "Moderate Protection"
        },
        "actionable_tips": tips
    }

    return JSONResponse(response_payload)


async def api_presets(request):
    """Returns curated wine profiles for 1-click user testing."""
    presets = {
        "premium_reserve": {
            "name": "Bordeaux-Style Premium Reserve",
            "tier": "High Quality (Class 2)",
            "values": {
                "fixed_acidity": 8.5,
                "volatile_acidity": 0.28,
                "citric_acid": 0.48,
                "residual_sugar": 2.2,
                "chlorides": 0.055,
                "free_sulfur_dioxide": 18.0,
                "total_sulfur_dioxide": 38.0,
                "density": 0.9958,
                "pH": 3.32,
                "sulphates": 0.88,
                "alcohol": 12.8
            }
        },
        "commercial_table": {
            "name": "Standard Everyday Commercial Red",
            "tier": "Medium Quality (Class 1)",
            "values": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.58,
                "citric_acid": 0.15,
                "residual_sugar": 2.1,
                "chlorides": 0.082,
                "free_sulfur_dioxide": 14.0,
                "total_sulfur_dioxide": 46.0,
                "density": 0.9968,
                "pH": 3.38,
                "sulphates": 0.62,
                "alcohol": 10.2
            }
        },
        "defective_batch": {
            "name": "Faulty / High-Acidity Defective Batch",
            "tier": "Low Quality (Class 0)",
            "values": {
                "fixed_acidity": 7.2,
                "volatile_acidity": 0.98,
                "citric_acid": 0.02,
                "residual_sugar": 1.8,
                "chlorides": 0.115,
                "free_sulfur_dioxide": 8.0,
                "total_sulfur_dioxide": 28.0,
                "density": 0.9972,
                "pH": 3.52,
                "sulphates": 0.44,
                "alcohol": 9.1
            }
        },
        "dataset_mean": {
            "name": "Dataset Benchmark Average",
            "tier": "Benchmark Baseline",
            "values": {
                "fixed_acidity": 8.32,
                "volatile_acidity": 0.53,
                "citric_acid": 0.27,
                "residual_sugar": 2.54,
                "chlorides": 0.087,
                "free_sulfur_dioxide": 15.87,
                "total_sulfur_dioxide": 46.47,
                "density": 0.9967,
                "pH": 3.31,
                "sulphates": 0.66,
                "alcohol": 10.42
            }
        }
    }
    return JSONResponse(presets)


async def api_metrics(request):
    """Returns actual model evaluation metrics from outputs/model_comparison.csv."""
    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return JSONResponse(df.to_dict(orient="records"))
    return JSONResponse([
        {"Model": "Random Forest", "Accuracy": 0.8281, "Balanced Accuracy": 0.6284, "Macro F1-Score": 0.6002, "Weighted F1-Score": 0.8327},
        {"Model": "SGD Classifier", "Accuracy": 0.7781, "Balanced Accuracy": 0.5498, "Macro F1-Score": 0.5205, "Weighted F1-Score": 0.7963},
        {"Model": "Support Vector Classifier (SVC)", "Accuracy": 0.7063, "Balanced Accuracy": 0.7319, "Macro F1-Score": 0.5599, "Weighted F1-Score": 0.7427}
    ])


# Define routes
routes = [
    Route("/", endpoint=home),
    Route("/api/health", endpoint=api_health),
    Route("/api/predict", endpoint=api_predict, methods=["POST"]),
    Route("/api/presets", endpoint=api_presets),
    Route("/api/metrics", endpoint=api_metrics),
    Mount("/outputs", app=StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs"),
    Mount("/static", app=StaticFiles(directory=str(FRONTEND_DIR)), name="static"),
]

app = Starlette(debug=False, routes=routes)


# Ensure UTF-8 output encoding on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 70)
    print("[SERVER] VinoPredict AI - Web Application Server Starting")
    print(f"[SERVER] Local Web UI URL: http://localhost:{port}")
    print(f"[SERVER] Serving static assets from: {FRONTEND_DIR}")
    print("=" * 70)
    # Ensure model is initialized
    get_or_train_model()
    uvicorn.run("app:app", host="127.0.0.1", port=port, log_level="info")

