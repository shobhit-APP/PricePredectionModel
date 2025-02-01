import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('Cropprice.csv')

# Load the scalers and model
with open('minmaxscaler.pkl', 'rb') as minmax_file:
    mx = pickle.load(minmax_file)

with open('standscaler.pkl', 'rb') as stand_file:
    sc = pickle.load(stand_file)

with open('model.pkl', 'rb') as model_file:
    randclf = pickle.load(model_file)
# Handle missing values

df.ffill(inplace=True)
# Crop dictionary to map numbers back to crop names
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

# Define the recommendation function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = mx.transform(features)
    features_standardized = sc.transform(features_scaled)
    prediction = randclf.predict(features_standardized)
    predicted_class = int(prediction[0])
    crop_name = crop_dict.get(predicted_class, "Unknown crop")
    return crop_name

# Fit Label Encoders comprehensively
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

# Train the models (XGBoost and Neural Network)
X = df[['state', 'district', 'market', 'crop_name', 'min_price', 'max_price']]  # Exclude 'arrival_date'
y = df['suggested_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
xgb_model.fit(X_train, y_train)
pickle.dump(xgb_model, open('cropPricePredictionModel.pkl', 'wb'))

nn_model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))
nn_model.save('nn_model.keras')  # Save model in Keras native format
@app.route('/')
def home():
    return "Welcome to the Crop Recommendation API!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not all(key in data for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']):
        return jsonify({'error': 'Missing input data'}), 400

    N = data['N']
    P = data['P']
    K = data['K']
    temperature = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']

    prediction = recommendation(N, P, K, temperature, humidity, ph, rainfall)

    return jsonify({'predicted_crop': prediction})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        new_data = pd.DataFrame({
            'state': [data['state']],
            'district': [data['district']],
            'market': [data['market']],
            'crop_name': [data['crop_name']],
            'min_price': [data['min_price']],
            'max_price': [data['max_price']]
        })
        
        # Ensure data types are correct
        new_data['min_price'] = new_data['min_price'].astype(float)
        new_data['max_price'] = new_data['max_price'].astype(float)

        # Handling unseen labels by assigning a default encoding
        def encode_column(column_name, encoder):
            try:
                return encoder.transform(new_data[column_name])
            except ValueError:
                # Add new classes to the encoder
                unique_values = list(encoder.classes_) + list(new_data[column_name].unique())
                encoder.classes_ = np.array(unique_values)
                return encoder.transform(new_data[column_name])
        
        try:
            new_data['state'] = encode_column('state', state_encoder)
            new_data['district'] = encode_column('district', district_encoder)
            new_data['market'] = encode_column('market', market_encoder)
            new_data['crop_name'] = encode_column('crop_name', crop_name_encoder)
        except ValueError as e:
            return jsonify({'error': f'Encoding error: {str(e)}'}), 400

        predicted_price_xgb = xgb_model.predict(new_data)
        predicted_price_xgb = float(predicted_price_xgb[0])
        predicted_price_nn = nn_model.predict(new_data)
        predicted_price_nn = float(predicted_price_nn[0][0])

        return jsonify({
            'predicted_price_xgb': predicted_price_xgb,
            'predicted_price_nn': predicted_price_nn
        })
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
