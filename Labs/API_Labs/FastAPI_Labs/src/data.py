import os
import pandas as pd
from sklearn.model_selection import train_test_split

# paths are built from this file's location, so the scripts work from any folder
LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(LAB_DIR, "data", "penguins.csv")
MODEL_PATH = os.path.join(LAB_DIR, "model", "penguin_model.pkl")

TARGET_COLUMN = "species"
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET_COLUMN])
    return df[FEATURES], df[TARGET_COLUMN]


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=12)
    return X_train, X_test, y_train, y_test
