import os
from datetime import datetime, timezone
import joblib
import sklearn
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from data import load_data, split_data, CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURES, MODEL_PATH


def fit_model(X_train, y_train):
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("passthrough", NUMERIC_FEATURES),
        verbose_feature_names_out=False,
    )
    model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=200, random_state=12))
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")
    return float(accuracy)


def save_model(model, test_accuracy, train_rows, test_rows):
    forest = model[-1]
    importances = dict(zip(model[0].get_feature_names_out(), forest.feature_importances_))
    # saved next to the model so the api can return it from /model-info
    info = {
        "algorithm": type(forest).__name__,
        "classes": [str(c) for c in model.classes_],
        "features": FEATURES,
        "test_accuracy": round(test_accuracy, 4),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "feature_importances": {
            name: round(float(value), 4)
            for name, value in sorted(importances.items(), key=lambda item: item[1], reverse=True)
        },
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "info": info}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    test_accuracy = evaluate_model(model, X_test, y_test)
    save_model(model, test_accuracy, len(X_train), len(X_test))
