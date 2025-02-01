
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load dataset
df = pd.read_csv('Cropprice.csv')

# Fill missing values
df.ffill(inplace=True)

# Define crop dictionary
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

# ✅ Train & Save Label Encoders
def fit_label_encoders(df, column_name, additional_values=[]):
    le = LabelEncoder()
    unique_values = list(df[column_name].unique()) + additional_values
    le.fit(unique_values)
    df[column_name] = le.transform(df[column_name])
    pickle.dump(le, open(f'{column_name}_encoder.pkl', 'wb'))
    return le

state_encoder = fit_label_encoders(df, 'state', ['Uttar Pradesh', 'Karnataka'])
district_encoder = fit_label_encoders(df, 'district', ['Basti', 'Shimoga'])
market_encoder = fit_label_encoders(df, 'market', ['Local Market', 'Shimoga Market'])
crop_name_encoder = fit_label_encoders(df, 'crop_name', ['Wheat'])

# ✅ Train & Save MinMaxScaler on correct 6 features
scaler_features = ['state', 'district', 'market', 'crop_name', 'min_price', 'max_price']
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[scaler_features])
pickle.dump(scaler, open('minmaxscaler.pkl', 'wb'))

# ✅ Train XGBoost Model
X_train, X_test, y_train, y_test = train_test_split(df[scaler_features], df['suggested_price'], test_size=0.2, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
xgb_model.fit(X_train, y_train)
pickle.dump(xgb_model, open('cropPricePredictionModel.pkl', 'wb'))

# ✅ Train Neural Network Model
nn_model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))
nn_model.save('nn_model.keras')

# ✅ Crop Recommendation Function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    with open('minmaxscaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    with open('model.pkl', 'rb') as model_file:
        randclf = pickle.load(model_file)
    
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)
    
    prediction = randclf.predict(features_scaled)
    predicted_class = int(prediction[0])
    return crop_dict.get(predicted_class, "Unknown crop")

# ✅ Handle Encoding of Input Data
def encode_column(data, column_name, encoder):
    try:
        return encoder.transform([data[column_name]])[0]
    except ValueError:
        unique_values = list(encoder.classes_) + [data[column_name]]
        encoder.classes_ = np.array(unique_values)
        return encoder.transform([data[column_name]])[0]

# ✅ Retrain Function (Optional API for Dynamic Model Updates)
@app.route('/retrain', methods=['POST'])
def retrain_model():
    global xgb_model, nn_model, scaler

    # Reload dataset & preprocess
    df = pd.read_csv('Cropprice.csv')
    df.ffill(inplace=True)
    
    # Retrain MinMaxScaler
    df_scaled = scaler.fit_transform(df[scaler_features])
    pickle.dump(scaler, open('minmaxscaler.pkl', 'wb'))

    # Retrain Models
    X_train, X_test, y_train, y_test = train_test_split(df[scaler_features], df['suggested_price'], test_size=0.2, random_state=42)
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    xgb_model.fit(X_train, y_train)
    pickle.dump(xgb_model, open('cropPricePredictionModel.pkl', 'wb'))

    nn_model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))
    nn_model.save('nn_model.keras')

    return jsonify({'message': 'Model retrained successfully'}), 200

# ✅ Flask Routes
@app.route('/')
def home():
    return "Welcome to the Crop Prediction API!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    
    required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing input data'}), 400

    prediction = recommendation(data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall'])
    
    return jsonify({'predicted_crop': prediction})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Encode categorical values
        new_data = pd.DataFrame({
            'state': [encode_column(data, 'state', state_encoder)],
            'district': [encode_column(data, 'district', district_encoder)],
            'market': [encode_column(data, 'market', market_encoder)],
            'crop_name': [encode_column(data, 'crop_name', crop_name_encoder)],
            'min_price': [float(data['min_price'])],
            'max_price': [float(data['max_price'])]
        })

        # Load Scaler & Models
        with open('minmaxscaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('cropPricePredictionModel.pkl', 'rb') as model_file:
            xgb_model = pickle.load(model_file)
        nn_model = load_model('nn_model.keras')

        # Scale Data
        new_data_scaled = scaler.transform(new_data)

        # Predict
        predicted_price_xgb = float(xgb_model.predict(new_data_scaled)[0])
        predicted_price_nn = float(nn_model.predict(new_data_scaled)[0][0])

        return jsonify({'predicted_price_xgb': predicted_price_xgb, 'predicted_price_nn': predicted_price_nn})

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Run Flask
if __name__ == '__main__':
    app.run(debug=True)
