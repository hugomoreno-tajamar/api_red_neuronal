import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import encode_features
import joblib


# Cargar el modelo
MODEL_PATH = "model/real_estate_price_predictor_20250325_170903.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Cargar preprocesadores (si los usaste en entrenamiento)
scaler_price = joblib.load("scalers/scaler_price.pkl") # MinMaxScaler para el precio

# Crear la app Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener los datos JSON
        data = request.get_json()
        
        # Extraer los valores en el orden correcto
        rooms = int(data["rooms"])
        baths = int(data["baths"])
        sqft = int(data["sqft"])
        typology = data["typology"]  # Esto debe manejarse con one-hot encoding si es categórico
        neighborhood = data["neighborhood"]  # También puede necesitar encoding
        
        # Crear el array de entrada
        input_data = np.array([[rooms, baths, sqft, typology, neighborhood]])

        input_df = pd.DataFrame(input_data, columns=["rooms", "baths", "sqft", "typology", "neighborhood"])
        
        # Normalizar los datos
        input_data_scaled = encode_features(input_df)
        
        # Hacer la predicción
        pred_scaled = model.predict(input_data_scaled)
        
        # Desnormalizar la predicción
        pred_price = np.expm1(scaler_price.inverse_transform(pred_scaled.reshape(-1, 1)))[0][0]
        
        # Devolver la respuesta
        return jsonify({"predicted_price": str(pred_price)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
