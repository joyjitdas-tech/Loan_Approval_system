import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model/model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model/scaler.pkl")
OHE_PATH = os.path.join(BASE_DIR, "model/ohe.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model/columns.pkl")