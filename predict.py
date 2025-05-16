import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import logging
import os
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='/root/crop-prediction/app.log', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = '/root/crop-prediction'
DATA_DIR = os.path.join(BASE_DIR, 'Predict')
MODEL_DIR = os.path.join(BASE_DIR, 'Models')

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
CSV_PATH = '/root/crop-prediction/Predict/Cropprice.csv'
price_minmax_path = os.path.join(MODEL_DIR, 'price_minmax_scaler.pkl')
price_stand_path = os.path.join(MODEL_DIR, 'price_standard_scaler.pkl')
xgb_model_path = os.path.join(MODEL_DIR, 'xgb_price_model.pkl')
nn_model_path = os.path.join(MODEL_DIR, 'nn_price_model.keras')

# Check CSV file
logger.debug(f"Checking CSV at: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    logger.error(f"CSV file not found: {CSV_PATH}")
    raise FileNotFoundError(f"Cropprice.csv not found at {CSV_PATH}")
if not os.access(CSV_PATH, os.R_OK):
    logger.error(f"CSV file not readable: {CSV_PATH}")
    raise PermissionError(f"Cropprice.csv not readable at {CSV_PATH}")

logger.info(f"Loading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Handle missing values
logger.info("Handling missing values")
df = df.ffill().bfill()

# Handle outliers
def remove_outliers(df, columns):
    df_cleaned = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            logger.info(f"Removed outliers from {col}: {len(df) - len(df_cleaned)} rows")
    return df_cleaned

price_columns = ['min_price', 'max_price', 'suggested_price']
price_columns_present = [col for col in price_columns if col in df.columns]
if price_columns_present:
    df = remove_outliers(df, price_columns_present)

# Encode categorical columns
label_encoders = {}
categorical_columns = ['state', 'district', 'market', 'crop_name']
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        encoder_path = os.path.join(MODEL_DIR, f'{col}_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(le, f)
        logger.info(f"Saved encoder for {col} at {encoder_path}")

# Train price prediction models
price_target_col = None
for col in ['suggested_price', 'modal_price']:
    if col in df.columns:
        price_target_col = col
        break

if price_target_col:
    logger.info(f"Training models with target: {price_target_col}")
    feature_cols = [col for col in categorical_columns if col in df.columns]
    if 'min_price' in df.columns:
        feature_cols.append('min_price')
    if 'max_price' in df.columns:
        feature_cols.append('max_price')

    X_price = df[feature_cols]
    y_price = df[price_target_col]

    # Scale features
    mx_price = MinMaxScaler()
    X_price_scaled = mx_price.fit_transform(X_price)
    sc_price = StandardScaler()
    X_price_standardized = sc_price.fit_transform(X_price_scaled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_price_standardized, y_price, test_size=0.2, random_state=42)

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    logger.info(f"XGBoost MSE: {mean_squared_error(y_test, xgb_pred)}, R2: {r2_score(y_test, xgb_pred)}")

    # Train Neural Network
    nn_model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
                 callbacks=[early_stopping], verbose=0)
    nn_pred = nn_model.predict(X_test, verbose=0)
    logger.info(f"Neural Network MSE: {mean_squared_error(y_test, nn_pred)}, R2: {r2_score(y_test, nn_pred.flatten())}")

    # Save models and scalers
    with open(price_minmax_path, 'wb') as f:
        pickle.dump(mx_price, f)
    with open(price_stand_path, 'wb') as f:
        pickle.dump(sc_price, f)
    with open(xgb_model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    nn_model.save(nn_model_path)
    logger.info("Models and scalers saved")
else:
    logger.warning("No price target column found. Skipping model training.")

# Load models and encoders
def load_price_models():
    try:
        with open(price_minmax_path, 'rb') as f:
            mx_price = pickle.load(f)
        with open(price_stand_path, 'rb') as f:
            sc_price = pickle.load(f)
        with open(xgb_model_path, 'rb') as f:
            xgb_model = pickle.load(f)
        nn_model = load_model(nn_model_path)
        return mx_price, sc_price, xgb_model, nn_model
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def load_encoders():
    encoders = {}
    for col in categorical_columns:
        encoder_path = os.path.join(MODEL_DIR, f'{col}_encoder.pkl')
        try:
            with open(encoder_path, 'rb') as f:
                encoders[col] = pickle.load(f)
        except:
            encoders[col] = None
    return encoders

def encode_column_safely(column_name, encoder, value):
    if encoder is None:
        return 0
    try:
        return encoder.transform([value])[0]
    except:
        logger.warning(f"Unseen value in {column_name}: {value}")
        return 0

def validate_price(predicted_price, min_price, max_price):
    if predicted_price < 0:
        return max(0, min_price * 0.8)
    if predicted_price > max_price * 2:
        return max_price * 1.2
    return predicted_price

@app.route('/')
def home():
    return "Crop Price Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")

        mx_price, sc_price, xgb_model, nn_model = load_price_models()
        if not all([mx_price, sc_price, xgb_model, nn_model]):
            return jsonify({'error': 'Models not available'}), 500
        
        encoders = load_encoders()
        if not any(encoders.values()):
            return jsonify({'error': 'Encoders not available'}), 500

        encoded_data = {}
        for col in categorical_columns:
            if col in data and encoders.get(col):
                encoded_data[col] = encode_column_safely(col, encoders[col], data[col])
            else:
                encoded_data[col] = 0

        min_price = float(data.get('min_price', 0))
        max_price = float(data.get('max_price', 0))
        encoded_data['min_price'] = min_price
        encoded_data['max_price'] = max_price

        input_array = np.array([[
            encoded_data.get('state', 0),
            encoded_data.get('district', 0),
            encoded_data.get('market', 0),
            encoded_data.get('crop_name', 0),
            encoded_data.get('min_price', 0),
            encoded_data.get('max_price', 0)
        ]])

        input_scaled = mx_price.transform(input_array)
        input_standardized = sc_price.transform(input_scaled)

        xgb_pred = float(xgb_model.predict(input_standardized)[0])
        nn_pred = float(nn_model.predict(input_standardized, verbose=0)[0][0])

        xgb_pred = validate_price(xgb_pred, min_price, max_price)
        nn_pred = validate_price(nn_pred, min_price, max_price)
        ensemble_pred = (xgb_pred + nn_pred) / 2

        result = {
            'predicted_price_xgb': round(xgb_pred, 2),
            'predicted_price_nn': round(nn_pred, 2),
            'predicted_price_ensemble': round(ensemble_pred, 2),
            'min_input_price': min_price,
            'max_input_price': max_price
        }
        logger.info(f"Prediction: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
