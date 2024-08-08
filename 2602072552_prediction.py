# Nama : Alicia Jocelyn Siahaya
# NIM : 2602072552

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Memanggil model SVM dan semua encoder yang sudah di pickle dari training
model = joblib.load('SVM_model.pkl')
categorical_encoder = joblib.load('categorical_encode.pkl')
day_encoder = joblib.load('day_encode.pkl')
month_encoder = joblib.load('month_encode.pkl')
default_encoder = joblib.load('default_encode.pkl')
scaler = joblib.load('scaler.pkl')

# Mendefinisikan data dari model yang menyatakan expected input data
class FailureType(BaseModel):
    age: float
    default: int
    month: int
    day_of_week: int
    duration: float
    campaign: float
    pdays: float
    previous: float
    job: str
    marital: str
    education: str
    housing: str
    loan: str
    contact: str
    poutcome: str

# Mendefinisikan Endpoint Root
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

# Mendefinisikan Endpoint Prediction
@app.post('/predict')
def predict(Failure: FailureType):
    data = Failure.dict() # Mengkonversi input menjadi dictionary
    
    # Menggabungkan data input yang memerlukan encoding
    categorical_features = np.array([[data['job'], data['marital'], data['education'], data['housing'], data['loan'], data['contact'], data['poutcome']]])
    day_feature = np.array([[data['day_of_week']]])
    month_feature = np.array([[data['month']]])
    default_feature = np.array([[data['default']]])
    numeric_features = np.array([[data['age'], data['duration'], data['campaign'], data['pdays'], data['previous']]])
    
    # Menggunakan encoder untuk mengubah data input
    encoded_categorical = categorical_encoder.transform(categorical_features)
    encoded_day = day_encoder.transform(day_feature)
    encoded_month = month_encoder.transform(month_feature)
    encoded_default = default_encoder.transform(default_feature)
    scaled_numeric = scaler.transform(numeric_features)
    
    # Menggabungkan semua fitur yang telah di-encode dan di-scale
    features = np.hstack([scaled_numeric, encoded_default, encoded_month, encoded_day, encoded_categorical])
    
    # Melakukan prediksi dengan model yang sudah dipanggil sebelumnya dari pickle (SVM_model)
    prediction = model.predict(features)
    
    return {'prediction': prediction[0]} # Mengeluarkan output hasil prediksi berdasarkan data input yang diberikan
