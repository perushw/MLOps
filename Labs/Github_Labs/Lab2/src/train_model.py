import mlflow, datetime, os, pickle
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import argparse
import sys

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # Load the Breast Cancer Wisconsin dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Save the test split so evaluate_model.py can score the model on it
    os.makedirs('data', exist_ok=True)
    with open('data/X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)
    with open('data/y_test.pickle', 'wb') as f:
        pickle.dump(y_test, f)

    # Set up MLflow experiment tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    dataset_name = "Breast Cancer in Wisconsin"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):

        params = {
            "dataset_name": dataset_name,
            "number of datapoints": X.shape[0],
            "number of dimensions": X.shape[1],
            "train size": X_train.shape[0],
            "test size": X_test.shape[0],
            "preprocessing": "StandardScaler",
            "model": "LogisticRegression",
        }
        mlflow.log_params(params)

        # Preprocessing + model in one pipeline:
        # the features range from under 1 to over 1000, so they are standardized
        # (mean 0, standard deviation 1) before reaching the classifier.
        # The scaler is fit on the training data only and saved together with the model.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=0)),
        ])
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mlflow.log_metrics({
            "Train Accuracy": accuracy_score(y_train, y_train_pred),
            "Test Accuracy": accuracy_score(y_test, y_test_pred),
            "Test F1 Score": f1_score(y_test, y_test_pred),
        })

        if not os.path.exists('models/'):
            os.makedirs("models/")

        # Save the model with a timestamp-based version; the workflow moves it into models/
        model_version = f'model_{timestamp}'
        model_filename = f'{model_version}_dt_model.joblib'
        dump(model, model_filename)