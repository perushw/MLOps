import joblib
import pandas as pd
from data import FEATURES, MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def predict_data(model, rows):
    X = pd.DataFrame(rows, columns=FEATURES)
    classes = [str(c) for c in model.classes_]
    results = []
    for probs in model.predict_proba(X):
        best = int(probs.argmax())
        results.append({
            "species": classes[best],
            "confidence": float(probs[best]),
            "probabilities": {name: float(p) for name, p in zip(classes, probs)},
        })
    return results
