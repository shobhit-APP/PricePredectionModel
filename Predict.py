import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from sklearn.utils.validation import check_array
from memory_profiler import profile
import logging
import os
import gc

app = Flask(__name__)

# Adjust the path according to the actual location of Cropprice.csv
file_path = os.path.join('Predict', 'Cropprice.csv')
df = pd.read_csv(file_path)

df.ffill(inplace=True)  # Handle missing values

# Adjust paths according to your project structure
minmax_path = os.path.join('Predict', 'minmaxscaler.pkl')
stand_path = os.path.join('Predict', 'standscaler.pkl')
model_path = os.path.join('Predict', 'model.pkl')
xgb_model_path = os.path.join('Predict', 'cropPricePredictionModel.pkl')
nn_model_path = os.path.join('Predict', 'nn_model.keras')

# Encode categorical columns using LabelEncoder
label_encoders = {}
categorical_columns = ['state', 'district', 'market', 'crop_name']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encode categorical values
    label_encoders[col] = le
    pickle.dump(le, open(f'Predict/{col}_encoder.pkl', 'wb'))  # Save encoders for later use

# Fit MinMaxScaler after encoding categorical columns
X_features = df[['state', 'district', 'market', 'crop_name', 'min_price', 'max_price']]
mx = MinMaxScaler()
mx.fit(X_features)
pickle.dump(mx, open(minmax_path, 'wb'))

# Fit StandardScaler with only the 6 features (state, district, market, crop_name, min_price, max_price)
sc = StandardScaler()
sc.fit(X_features)  # Fit using the 6 features
pickle.dump(sc, open(stand_path, 'wb'))

# Load Pretrained Models & Scalers
with open(minmax_path, 'rb') as minmax_file:
    mx = pickle.load(minmax_file)

with open(stand_path, 'rb') as stand_file:
    sc = pickle.load(stand_file)

with open(model_path, 'rb') as model_file:
    randclf = pickle.load(model_file)

xgb_model = pickle.load(open(xgb_model_path, 'rb'))
nn_model = load_model(nn_model_path)  # Load neural network model

# Train the models (XGBoost and Neural Network)
X = df[['state', 'district', 'market', 'crop_name','min_price', 'max_price']]
y = df['suggested_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
xgb_model.fit(X_train, y_train)
pickle.dump(xgb_model, open(xgb_model_path, 'wb'))

nn_model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
nn_model.save(nn_model_path)  # Save model in Keras native format

@app.route('/')
def home():
    return "Welcome to the Crop Recommendation API!"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Helper function to handle unseen labels
def encode_column_with_new_labels(column_name, encoder, new_data):
    try:
        return encoder.transform(new_data[column_name])
    except ValueError:
        # Handle unseen labels: add the new labels to the encoder and retrain it
        unique_values = list(encoder.classes_) + list(new_data[column_name].unique())
        encoder.classes_ = np.array(unique_values)
        pickle.dump(encoder, open(f'Predict/{column_name}_encoder.pkl', 'wb'))  # Save updated encoder
        return encoder.transform(new_data[column_name])

@app.route('/predict', methods=['POST'])
@profile
def predict():
    try:
        data = request.get_json()
        logging.info("Received Data: %s", data)  # Log received data

        # Prepare new input data as DataFrame
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
        logging.info("Data after initial processing: %s", new_data)

        # Encode categorical columns using the pre-trained label encoders (or update them)
        new_data['state'] = encode_column_with_new_labels('state', label_encoders['state'], new_data)
        new_data['district'] = encode_column_with_new_labels('district', label_encoders['district'], new_data)
        new_data['market'] = encode_column_with_new_labels('market', label_encoders['market'], new_data)
        new_data['crop_name'] = encode_column_with_new_labels('crop_name', label_encoders['crop_name'], new_data)

        logging.info("Data after encoding: %s", new_data)

        # Ensure the feature names are consistent with those used during fitting
        new_data_checked = check_array(new_data, dtype=np.float32, ensure_2d=True, allow_nd=False)
        new_data_scaled = mx.transform(new_data_checked)
        new_data_standardized = sc.transform(new_data_scaled)

        # Predictions using XGBoost and Neural Network models
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
