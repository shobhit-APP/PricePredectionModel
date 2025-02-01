import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)

# ✅ Define paths
minmax_path = os.path.join('Predict', 'minmaxscaler.pkl')
stand_path = os.path.join('Predict', 'standscaler.pkl')
model_path = os.path.join('Predict', 'model.pkl')
xgb_model_path = os.path.join('Predict', 'cropPricePredictionModel.pkl')
nn_model_path = os.path.join('Predict', 'nn_model.keras')

# ✅ Load Pretrained Models & Scalers
with open(minmax_path, 'rb') as minmax_file:
    mx = pickle.load(minmax_file)  # MinMaxScaler

with open(stand_path, 'rb') as stand_file:
    sc = pickle.load(stand_file)  # StandardScaler

with open(model_path, 'rb') as model_file:
    randclf = pickle.load(model_file)  # Random Forest Model (Unused in this script)

with open(xgb_model_path, 'rb') as xgb_file:
    xgb_model = pickle.load(xgb_file)  # XGBoost Model

nn_model = load_model(nn_model_path)  # Neural Network Model

# ✅ Load Label Encoders
def load_label_encoder(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

state_encoder = load_label_encoder(os.path.join('Predict', 'state_encoder.pkl'))
district_encoder = load_label_encoder(os.path.join('Predict', 'district_encoder.pkl'))
market_encoder = load_label_encoder(os.path.join('Predict', 'market_encoder.pkl'))
crop_name_encoder = load_label_encoder(os.path.join('Predict', 'crop_name_encoder.pkl'))
# ✅ Encoding Function
def encode_column(data, column_name, encoder):
    try:
        return encoder.transform([data[column_name]])[0]
    except ValueError:
        unique_values = list(encoder.classes_) + [data[column_name]]
        encoder.classes_ = np.array(unique_values)
        return encoder.transform([data[column_name]])[0]

# ✅ Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Convert categorical inputs into numerical form
        new_data = pd.DataFrame({
            'state': [encode_column(data, 'state', state_encoder)],
            'district': [encode_column(data, 'district', district_encoder)],
            'market': [encode_column(data, 'market', market_encoder)],
            'crop_name': [encode_column(data, 'crop_name', crop_name_encoder)],
            'min_price': [float(data['min_price'])],
            'max_price': [float(data['max_price'])]
        })

        # Use MinMaxScaler to scale the data
        new_data_scaled = mx.transform(new_data)

        # Predict using XGBoost and Neural Network
        predicted_price_xgb = float(xgb_model.predict(new_data_scaled)[0])
        predicted_price_nn = float(nn_model.predict(new_data_scaled)[0][0])

        return jsonify({'predicted_price_xgb': predicted_price_xgb, 'predicted_price_nn': predicted_price_nn})

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# ✅ Run Flask API
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
