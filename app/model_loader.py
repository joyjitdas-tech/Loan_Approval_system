import pickle
from app.config import MODEL_PATH, SCALER_PATH, OHE_PATH, COLUMNS_PATH

def load_artifacts():
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    ohe = pickle.load(open(OHE_PATH, "rb"))
    columns = pickle.load(open(COLUMNS_PATH, "rb"))

    return model, scaler, ohe, columns