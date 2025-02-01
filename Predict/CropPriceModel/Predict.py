import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)

# ✅ Load Pre-trained Models and Scalers
with open('minmaxscaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('cropPricePredictionModel.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

nn_model = load_model('nn_model.keras')

# ✅ Load Label Encoders
def load_label_encoder(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

state_encoder = load_label_encoder('state_encoder.pkl')
district_encoder = load_label_encoder('district_encoder.pkl')
market_encoder = load_label_encoder('market_encoder.pkl')
crop_name_encoder = load_label_encoder('crop_name_encoder.pkl')

# ✅ Handle Encoding of Input Data
def encode_column(data, column_name, encoder):
    try:
        return encoder.transform([data[column_name]])[0]
    except ValueError:
        unique_values = list(encoder.classes_) + [data[column_name]]
        encoder.classes_ = np.array(unique_values)
        return encoder.transform([data[column_name]])[0]

# ✅ Price Prediction Endpoint
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

        # Scale input data
        new_data_scaled = scaler.transform(new_data)

        # Predict using XGBoost and Neural Network
        predicted_price_xgb = float(xgb_model.predict(new_data_scaled)[0])
        predicted_price_nn = float(nn_model.predict(new_data_scaled)[0][0])

        return jsonify({'predicted_price_xgb': predicted_price_xgb, 'predicted_price_nn': predicted_price_nn})

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# ✅ Run Flask API
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
