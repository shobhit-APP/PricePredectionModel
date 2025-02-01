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
import logging
import os
from sklearn.utils.validation import check_array
from memory_profiler import profile
import gc

app = Flask(__name__)

# Adjust the path according to the actual location of Cropprice.csv
file_path = os.path.join('Model', 'Cropprice.csv')
df = pd.read_csv(file_path)

df.ffill(inplace=True)  # Handle missing values

# Adjust paths according to your project structure
minmax_path = os.path.join('Model', 'minmaxscaler.pkl')
stand_path = os.path.join('Model', 'standscaler.pkl')
model_path = os.path.join('Model', 'model.pkl')
xgb_model_path = os.path.join('Model', 'cropPricePredictionModel.pkl')
nn_model_path = os.path.join('Model', 'nn_model.keras')

# Load Pretrained Models & Scalers
with open(minmax_path, 'rb') as minmax_file:
    mx = pickle.load(minmax_file)

with open(stand_path, 'rb') as stand_file:
    sc = pickle.load(stand_file)

with open(model_path, 'rb') as model_file:
    randclf = pickle.load(model_file)

xgb_model = pickle.load(open(xgb_model_path, 'rb'))
nn_model = load_model(nn_model_path)  # Load neural network model



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
X = df[['state', 'district', 'market', 'crop_name','min_price', 'max_price','arrivalDate']]  
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
nn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
nn_model.save('nn_model.keras')  # Save model in Keras native format

@app.route('/')
def home():
    return "Welcome to the Crop Recommendation API!"

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
@profile
def predict():
    try:
        data = request.get_json()
        logging.info("Received Data: %s", data)  # Log received data

        new_data = pd.DataFrame({
            'state': [data['state']],
            'district': [data['district']],
            'market': [data['market']],
            'crop_name': [data['crop_name']],
            'min_price': [data['min_price']],
            'max_price': [data['max_price']]
        })

        new_data['min_price'] = new_data['min_price'].astype(float)
        new_data['max_price'] = new_data['max_price'].astype(float)
        logging.info("Data after initial processing: %s", new_data)  # Log processed data

        def encode_column(column_name, encoder):
            try:
                return encoder.transform(new_data[column_name])
            except ValueError:
                unique_values = list(encoder.classes_) + list(new_data[column_name].unique())
                encoder.classes_ = np.array(unique_values)
                return encoder.transform(new_data[column_name])

        new_data['state'] = encode_column('state', state_encoder)
        new_data['district'] = encode_column('district', district_encoder)
        new_data['market'] = encode_column('market', market_encoder)
        new_data['crop_name'] = encode_column('crop_name', crop_name_encoder)

        logging.info("Data after encoding: %s", new_data)  # Log encoded data

        # Ensure the feature names are consistent with those used during fitting
        new_data_checked = check_array(new_data, dtype=np.float32, ensure_2d=True, allow_nd=False)
        new_data_scaled = mx.transform(new_data_checked)
        new_data_standardized = sc.transform(new_data_scaled)

        predicted_price_xgb = xgb_model.predict(new_data_standardized)
        predicted_price_xgb = float(predicted_price_xgb[0])
        predicted_price_nn = nn_model.predict(new_data_standardized)
        predicted_price_nn = float(predicted_price_nn[0][0])

        # Run garbage collection to free up memory
        gc.collect()

        return jsonify({
            'predicted_price_xgb': predicted_price_xgb,
            'predicted_price_nn': predicted_price_nn
        })
    except ValueError as ve:
        logging.error("ValueError: %s", str(ve))  # Log specific error
        return jsonify({'error': f'ValueError: %s' % str(ve)}), 400
    except Exception as e:
        logging.error("Unexpected error: %s", str(e))  # Log unexpected error
        return jsonify({'error': f'Unexpected error: %s' % str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get PORT from Render, default to 5000
    logging.info("Running on port: %d", port)  # Debug statement
    app.run(host="0.0.0.0", port=port)
