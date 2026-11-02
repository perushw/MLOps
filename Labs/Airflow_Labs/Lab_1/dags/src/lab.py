import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import pickle
import os
import base64

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
# true cultivar label: dropped before training, only used to evaluate the clusters
LABEL_COLUMN = "cultivar"
K_RANGE = range(1, 11)


# load the wine data as a base64 string so it can be passed to the next task through xcom
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "wine.csv"))
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


# drop the label, standardize the features and reduce them to 2 principal components
def data_preprocessing(data_b64: str):
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    features = df.drop(columns=[LABEL_COLUMN])

    # 2 components keep the gmm's covariance matrices small enough to estimate from ~175 rows
    preprocessor = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=42))
    reduced_data = preprocessor.fit_transform(features)

    # pass the fitted preprocessor on so it can be saved with the model
    serialized_data = pickle.dumps((reduced_data, preprocessor))
    return base64.b64encode(serialized_data).decode("ascii")


# fit a gmm for each k, keep the one with the lowest bic and save it
def build_save_model(data_b64: str, filename: str):
    data_bytes = base64.b64decode(data_b64)
    reduced_data, preprocessor = pickle.loads(data_bytes)

    gmm_kwargs = {"covariance_type": "full", "n_init": 5, "random_state": 42}
    bic = []
    best_gmm = None
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, **gmm_kwargs)
        gmm.fit(reduced_data)
        bic.append(float(gmm.bic(reduced_data)))
        if bic[-1] == min(bic):
            best_gmm = gmm

    # save the preprocessor and gmm together so the model accepts raw feature rows
    model = make_pipeline(preprocessor, best_gmm)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(model, f)

    return {"k_values": list(K_RANGE), "bic": bic, "best_k": int(best_gmm.n_components)}


# report the chosen k, score the clusters against the true cultivars and predict clusters for the test wines
def load_model_evaluate(filename: str, results: dict):
    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        loaded_model = pickle.load(f)

    for k, score in zip(results["k_values"], results["bic"]):
        print(f"k={k:2d}  BIC={score:.2f}")
    print(f"Optimal no. of clusters (lowest BIC): {results['best_k']}")

    df = pd.read_csv(os.path.join(DATA_DIR, "wine.csv")).dropna()
    train_clusters = loaded_model.predict(df.drop(columns=[LABEL_COLUMN]))
    print(f"Adjusted Rand Index vs. true cultivars: {adjusted_rand_score(df[LABEL_COLUMN], train_clusters):.3f}")

    test_df = pd.read_csv(os.path.join(DATA_DIR, "wine_test.csv"))
    preds = [int(p) for p in loaded_model.predict(test_df)]
    probs = loaded_model.predict_proba(test_df)
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        print(f"Test row {i}: cluster {pred} (probability {prob.max():.3f})")

    return preds
