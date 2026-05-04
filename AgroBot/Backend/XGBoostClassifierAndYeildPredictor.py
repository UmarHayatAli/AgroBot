# main.py - AgroBot Soil-to-Crop Suitability Matrix
import pandas as pd
import pickle
import warnings
import json
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np

warnings.filterwarnings('ignore')

# ====================== LOAD MODELS ======================
print("[AgroBot] Loading models...")
model_path = os.path.join(os.path.dirname(__file__), "Models", "xgboost_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "Models", "scaler.pkl")
le_path = os.path.join(os.path.dirname(__file__), "Models", "label_encoder.pkl")
knn_path = os.path.join(os.path.dirname(__file__), "Models", "knn_model.pkl")
explainer_path = os.path.join(os.path.dirname(__file__), "Models", "shap_explainer.pkl")
xgb_model = pickle.load(open(model_path, 'rb'))
scaler     = pickle.load(open(scaler_path, 'rb'))
le         = pickle.load(open(le_path, 'rb'))
knn_model  = pickle.load(open(knn_path, 'rb'))
explainer  = pickle.load(open(explainer_path, 'rb'))

print("[AgroBot] Classifier models loaded successfully!\n")

# ====================== FEATURE RANGES (from your original code) ======================
FEATURE_RANGES = {
    'N':          (0,  140),
    'P':          (5,  145),
    'K':          (5,  205),
    'temperature': (8,  43),
    'humidity':   (10, 100),
    'ph':         (3.5, 9.5),
    'rainfall':   (20, 300),
    'Temperature': (8,  43),
    'Humidity':   (10, 100),
    'pH':         (3.5, 9.5),
    'Rainfall':   (20, 300),
    'EC':         (0, 5)
}
def normalize_keys(d):
    mapping = {
        'Temperature': 'temperature',
        'temperature': 'temperature',
        'Humidity': 'humidity',
        'humidity': 'humidity',
        'Rainfall': 'rainfall',
        'rainfall': 'rainfall',
        'pH': 'ph',
        'ph': 'ph'
    }
    
    return {mapping.get(k, k): v for k, v in d.items()}
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj
def validate_crop_input(user_input: dict):
    warnings_list = []
    for feature, value in user_input.items():
        key = feature.lower() if feature.lower() in ['temperature', 'humidity', 'ph', 'rainfall'] else feature
        if key in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[key]
            if value is None:
                continue
            if value < min_val or value > max_val:
                warnings_list.append(f"• {feature} = {value} is outside realistic range ({min_val} – {max_val})")

    temp = user_input.get('temperature') or user_input.get('Temperature')
    if temp is not None:
        if temp > 40:
            warnings_list.append(f"• Temperature = {temp}°C is EXTREMELY HIGH. Most crops will suffer severe heat stress.")
        elif temp > 37:
            warnings_list.append(f"• Temperature = {temp}°C is very high and risky for many crops.")

    return warnings_list

# ====================== REALISTIC BASE YIELDS (tons per acre) ======================
BASE_YIELD = {
    'rice':        2.85,
    'wheat':       2.65,
    'maize':       3.80,
    'soybean':     2.10,
    'chickpea':    1.45,
    'kidneybeans': 1.35,
    'pigeonpeas':  1.25,
    'mothbeans':   1.10,
    'mungbean':    1.40,
    'blackgram':   1.30,
    'lentil':      1.40,
    'cotton':      2.20,
    'jute':        2.50,
    'coffee':      0.85,
    'apple':       8.50,
    'orange':      12.00,
    'papaya':      15.00,
    'coconut':     80.00,   # nuts per acre (special handling)
    'muskmelon':   8.50,
}

DEFAULT_BASE_YIELD = 2.5

# ====================== YIELD ADJUSTMENT FUNCTION ======================
def calculate_yield_adjustment(user_input: dict, crop: str) -> float:
    adjustment = 0.0
    crop_lower = crop.lower()
    rain = float(user_input.get('Rainfall') or user_input.get('rainfall') or 150)
    temp = float(user_input.get('Temperature') or user_input.get('temperature') or 25)
    ph = float(user_input.get('pH') or user_input.get('ph') or 6.5)
    # Soil Type
    soil = str(user_input.get('Soil_Type', 'Loamy')).lower()
    if soil == 'loamy':
        adjustment += 0.25
    elif soil in ['silty', 'clayey']:
        adjustment += 0.15
    elif soil == 'sandy':
        adjustment -= 0.20

    # Salinity (EC) - Critical for Pakistan
    ec = float(user_input.get('EC') or 1.0)
    if ec > 2.5:
        adjustment -= 0.45
    elif ec > 1.8:
        adjustment -= 0.30
    elif ec > 1.2:
        adjustment -= 0.15

    # NPK Balance
    n = float(user_input.get('N', 80))
    p = float(user_input.get('P', 40))
    k = float(user_input.get('K', 40))
    total_npk = n + p + k
    if total_npk > 220:
        adjustment += 0.25
    elif total_npk > 160:
        adjustment += 0.12
    elif total_npk < 90:
        adjustment -= 0.30

    # Rainfall
    
    if crop_lower in ['rice', 'papaya', 'coconut']:
        if rain > 200: adjustment += 0.30
        elif rain < 100: adjustment -= 0.35
    else:
        if rain > 250: adjustment -= 0.15
        elif rain < 80: adjustment -= 0.25

    # Temperature
    if temp > 38: adjustment -= 0.35
    elif temp > 35: adjustment -= 0.20
    elif temp < 12: adjustment -= 0.25

    # pH
    if ph < 5.5 or ph > 8.5:
        adjustment -= 0.20

    return round(adjustment, 3)


# ====================== MAIN INTELLIGENT PREDICTION FUNCTION ======================
def agro_predict(user_input: dict):
    user_input = normalize_keys(user_input)
    def coerce_numeric(d):
     for k, v in d.items():
        try:
            if v is not None:
                d[k] = float(v)
        except:
            d[k] = None
     return d

    user_input = coerce_numeric(user_input)


    validation_warnings = validate_crop_input(user_input)
    
    # Prepare input for Classifier
    clf_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    clf_input = {k: user_input.get(k) for k in clf_features}
    missing = [k for k, v in clf_input.items() if v is None]

    if missing:
     return {
        "error": f"Missing required features: {missing}",
        "received_input": user_input
    }

    # Classifier Prediction
    input_scaled = scaler.transform(pd.DataFrame([clf_input]))
    pred_encoded = xgb_model.predict(input_scaled)[0]
    pred_crop = le.inverse_transform([pred_encoded])[0]
    pred_proba = xgb_model.predict_proba(input_scaled)[0]
    confidence = round(float(pred_proba[pred_encoded] * 100), 1)
    feature_importance = {}
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(input_scaled)
            shap_for_pred = shap_values[0, :, pred_encoded] if len(shap_values.shape) == 3 else shap_values[0]
            feature_importance = {
                feature: round(float(value), 4)
                for feature, value in zip(clf_features, shap_for_pred)
            }
        except:
            feature_importance = {"note": "SHAP calculation failed"}
    # Yield Calculation (Rule-based)
    base_yield = BASE_YIELD.get(pred_crop.lower(), DEFAULT_BASE_YIELD)
    adjustment = calculate_yield_adjustment(user_input, pred_crop)
    predicted_yield = max(0.5, base_yield + adjustment)

    # Suitability
    if predicted_yield >= 4.0:
        suitability = "Excellent 🌟"
    elif predicted_yield >= 3.0:
        suitability = "Good 🟢"
    elif predicted_yield >= 2.0:
        suitability = "Moderate 🟡"
    else:
        suitability = "Low / Risky 🔴"
    
    # Similar Cases (from your original code)
    temp = user_input.get('temperature') or user_input.get('Temperature')
    rain = user_input.get('rainfall') or user_input.get('Rainfall')
    
    _csv_path = os.path.join(os.path.dirname(__file__), 'Crop_recommendation.csv')
    df_clean = pd.read_csv(_csv_path)  # Load for similar cases
    candidates = df_clean[
        (df_clean['temperature'].between(temp - 6, temp + 6)) &
        (df_clean['rainfall'].between(rain - 60, rain + 60))
    ].copy()

    if len(candidates) >= 3:
        X_cand_scaled = scaler.transform(candidates[clf_features])
        nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        nn.fit(X_cand_scaled)
        _, indices = nn.kneighbors(input_scaled)
        similar_df = candidates.iloc[indices[0]]
    else:
        _, indices = knn_model.kneighbors(input_scaled, n_neighbors=5)
        similar_df = df_clean.iloc[indices[0]]

    similar_cases = []
    for _, row in similar_df.iterrows():
        similar_cases.append({
            "N": int(row['N']),
            "P": int(row['P']),
            "K": int(row['K']),
            "temp": round(float(row['temperature']), 1),
            "humidity": round(float(row['humidity']), 1),
            "ph": round(float(row['ph']), 2),
            "rainfall": round(float(row['rainfall']), 1),
            "label": row['label'].lower()
        })

    # SHAP Feature Importance (optional - can be heavy)
    
    # explainer = ... (you can add if needed)

    result = {
        "recommended_crop": pred_crop.capitalize(),
        "predicted_yield_tons_per_acre": round(predicted_yield, 2),
        "suitability": suitability,
        "confidence": confidence,
        "base_yield": round(base_yield, 2),
        "adjustment": adjustment,
        "top_3_crops": [(le.classes_[i].capitalize(), round(pred_proba[i]*100, 1)) 
                       for i in pred_proba.argsort()[-3:][::-1]],
        "similar_cases": similar_cases[:5],
        "feature_importance": feature_importance,

        "validation_warnings": validation_warnings if validation_warnings else None,
        "note": "Yield estimated using crop-specific baseline + agronomic adjustments (Rule-based)"
    }

    return convert_to_serializable(result)


# ====================== TEST ======================
if __name__ == "__main__":
    print("=== AgroBot Soil-to-Crop Suitability Matrix ===\n")

    # Test 1: Full input with EC, Soil_Type
    test1 = {
        'N': 98.7, 'P': 5.27, 'K': 28.11,
        'Temperature': 21.23, 'Humidity': 52.42, 'Rainfall': 219,
        'pH': 6.94, 'EC': 0.89,
        'Soil_Type': 'Sandy',
        'Crop_Type': 'Maize'   # ignored in fallback mode
    }

    print("Test 1 - Full Input:")
    print(json.dumps(agro_predict(test1), indent=2))

    # Test 2: Partial input (only classifier features)
    test2 = {
        'N': 90, 'P': 42, 'K': 43,
        'temperature': 30, 'humidity': 82, 'ph': 6.5, 'rainfall': 202
    }

    print("\nTest 2 - Partial Input:")
    print(json.dumps(agro_predict(test2), indent=2))