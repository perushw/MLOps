import pickle, os, json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import joblib, sys
import argparse

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp

    try:
        model_version = f'model_{timestamp}_dt_model'
        model = joblib.load(f'{model_version}.joblib')
    except Exception as e:
        raise ValueError(f'Failed to load the latest model: {e}')

    try:
        # Load the held-out Breast Cancer test split saved by train_model.py
        with open('data/X_test.pickle', 'rb') as f:
            X = pickle.load(f)
        with open('data/y_test.pickle', 'rb') as f:
            y = pickle.load(f)
    except Exception as e:
        raise ValueError(f'Failed to load the test data: {e}')

    y_predict = model.predict(X)
    metrics = {
        "F1_Score": f1_score(y, y_predict),
        "Accuracy": accuracy_score(y, y_predict),
        "Precision": precision_score(y, y_predict),
        "Recall": recall_score(y, y_predict),
        "Test_Samples": int(len(y)),
    }

    if not os.path.exists('metrics/'):
        os.makedirs("metrics/")

    # Written to the current directory; the workflow moves it into metrics/
    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(json.dumps(metrics, indent=4))