from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schema import LoanRequest
from app.model_loader import load_artifacts
from app.predictor import predict_loan

app = FastAPI(title="Loan Approval API 🚀")

# Load model artifacts once
model, scaler, ohe, columns = load_artifacts()

# ✅ Serve static files (frontend)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ✅ Open frontend at root
@app.get("/")
def home():
    return FileResponse("app/static/index.html")

# ✅ Prediction API
@app.post("/predict")
def predict(data: LoanRequest):
    result = predict_loan(data.dict(), model, scaler, ohe, columns)
    return result