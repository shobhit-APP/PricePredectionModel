import pandas  import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from memory_profiler import profile
import logging
import os
import gc
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory of the current script file
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
 Franciscan
    BASE_DIR = os.getcwd()

# Data and Model directories
DATA_DIR = os.path.join(BASE_DIR, 'Predict')
MODEL_DIR = os.path.join(BASE_DIR, 'Models')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Debug path information
logger.debug(f"BASE_DIR: {BASE_DIR}")
logger.debug(f"DATA_DIR: {DATA_DIR}")
logger.debug(f"MODEL_DIR: {MODEL_DIR}")

# File paths to try (prioritize known correct path)
file_paths_to_try = [
    '/root/crop-prediction/Predict/Cropprice.csv',  # Known correct path
    os.path.join(DATA_DIR, 'Cropprice.csv'),
    os.path.join(BASE_DIR, 'Cropprice.csv'),
    './Predict/Cropprice.csv',
    './Cropprice.csv'
]

# Log resolved paths
logger.debug(f"Resolved file_paths_to_try: {file_paths_to_try}")

# Try each path until we find the file
file_path = None
for path in file_paths_to_try:
    logger.debug(f"Trying path: {path}")
    if os.path.exists(path) and os.access(path, os.R_OK):
        file_path = path
        logger.info(f"Found CSV at: {file_path}")
        break

# Exit if file not found or not readable
if file_path is None:
    logger.error(f"Cropprice.csv not found or not readable in any of these locations: {file_paths_to_try}")
    raise FileNotFoundError("Cropprice.csv not found or not readable")

# Define other file paths
price_minmax_path = os.path.join(MODEL_DIR, 'price_minmax_scaler.pkl')
price_stand_path = os.path.join(MODEL_DIR, 'price_standard_scaler.pkl')
xgb_model_path = os.path.join(MODEL_DIR, 'xgb_price_model.pkl')
nn_model_path = os.path.join(MODEL_DIR, 'nn_price_model.keras')

# Check if model and scaler files exist
for model_file in [price_minmax_path, price_stand_path, xgb_model_path, nn_model_path]:
    if not os.path.exists(model_file):
        logger.warning(f"Model/scaler file not found: {model_file}. Will attempt to train and save.")

logger.info(f"Loading dataset from: {file_path}")
try:
    df = pd.read_csv(file_path)
except Exception as e:
    logger.error(f"Failed to load CSV file: {str(e)}")
    raise

# Handle missing values
logger.info("Handling missing values")
df = df.fillna(method='ffill').fillna(method='bfill')  # Updated syntax

# Check for and handle outliers in price columns
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
else:
    logger.warning("No price columns found in dataset")

# Encode categorical columns and save encoders
label_encoders = {}
categorical_columns = ['state', 'district', 'market', 'crop_name']

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        encoder_path = os.path.join(MODEL_DIR, f'{col}_encoder.pkl')
        try:
            with open(encoder_path, 'wb') as f:
                pickle.dump(le, f)
            logger.info(f"Saved encoder for {col} at {encoder_path}")
        except Exception as e:
            logger.error(f"Failed to save encoder for {col}: {str(e)}")

# Create price prediction models if target exists
price_target_col = None
for col in ['suggested_price', 'modal_price']:
    if col in df.columns:
        price_target_col = col
        break

if price_target_col:
    logger.info(f"Training price prediction models using target: {price_target_col}")
    
    # Select relevant features for price prediction
    feature_cols = [col for col in categorical_columns if col in df.columns]
    
    # Add price features if they exist
    if 'min_price' in df.columns:
        feature_cols.append('min_price')
    if 'max_price' in df.columns:
        feature_cols.append('max_price')
    
    if not feature_cols:
        logger.error("No valid feature columns found for price prediction")
        raise ValueError("No valid feature columns")

    X_price = df[feature_cols]
    y_price = df[price_target_col]
    
    # Print some statistics about the price data
    logger.info(f"Price statistics - Min: {y_price.min()}, Max: {y_price.max()}, Mean: {y_price.mean()}, Median: {y_price.median()}")
    
    # Fit MinMaxScaler and StandardScaler for price prediction
    mx_price = MinMaxScaler()
    X_price_scaled = mx_price.fit_transform(X_price)
    
    sc_price = StandardScaler()
    X_price_standardized = sc_price.fit_transform(X_price_scaled)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_price_standardized, y_price, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost model
    xgb_pred = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    logger.info(f"XGBoost MSE: {xgb_mse}, R2: {xgb_r2}")
    
    # Save XGBoost model
    try:
        with open(xgb_model_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        logger.info(f"Saved XGBoost model at {xgb_model_path}")
    except Exception as e:
        logger.error(f"Failed to save XGBoost model: {str(e)}")
    
    # Train Neural Network model with improved architecture
    nn_model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    try:
        nn_history = nn_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
    except Exception as e:
        logger.error(f"Failed to train Neural Network model: {str(e)}")
        raise
    
    # Evaluate Neural Network model
    nn_pred = nn_model.predict(X_test)
    nn_mse = mean_squared_error(y_test, nn_pred)
    nn_r2 = r2_score(y_test, nn_pred.flatten())
    logger.info(f"Neural Network MSE: {nn_mse}, R2: {nn_r2}")
    
    # Save Neural Network model
    try:
        nn_model.save(nn_model_path)
        logger.info(f"Saved Neural Network model at {nn_model_path}")
    except Exception as e:
        logger.error(f"Failed to save Neural Network model: {str(e)}")
    
    # Save price scalers
    try:
        with open(price_minmax_path, 'wb') as f:
            pickle.dump(mx_price, f)
        logger.info(f"Saved MinMaxScaler at {price_minmax_path}")
    except Exception as e:
        logger.error(f"Failed to save MinMaxScaler: {str(e)}")
    
    try:
        with open(price_stand_path, 'wb') as f:
            pickle.dump(sc_price, f)
        logger.info(f"Saved StandardScaler at {price_stand_path}")
    except Exception as e:
        logger.error(f"Failed to save StandardScaler: {str(e)}")
    
    logger.info("Price prediction models and scalers saved successfully")
else:
    logger.warning("Price prediction target column (suggested_price or modal_price) not found in dataset")

# Load price models and scalers
def load_price_models():
    try:
        with open(price_minmax_path, 'rb') as f:
            mx_price = pickle.load(f)
        logger.debug(f"Loaded MinMaxScaler from {price_minmax_path}")
        
        with open(price_stand_path, 'rb') as f:
            sc_price = pickle.load(f)
        logger.debug(f"Loaded StandardScaler from {price_stand_path}")
        
        with open(xgb_model_path, 'rb') as f:
            xgb_model = pickle.load(f)
        logger.debug(f"Loaded XGBoost model from {xgb_model_path}")
        
        nn_model = load_model(nn_model_path)
        logger.debug(f"Loaded Neural Network model from {nn_model_path}")
        
        return mx_price, sc_price, xgb_model, nn_model
    except Exception as e:
        logger.error(f"Error loading price models: {str(e)}")
        return None, None, None, None

def load_encoders():
    encoders = {}
    for col in categorical_columns:
        encoder_path = os.path.join(MODEL_DIR, f'{col}_encoder.pkl')
        try:
            with open(encoder_path, 'rb') as f:
                encoders[col] = pickle.load(f)
            logger.debug(f"Loaded encoder for {col} from {encoder_path}")
        except Exception as e:
            logger.warning(f"Could not load encoder for {col}: {str(e)}")
    return encoders

# Helper function to handle unseen labels safely
def encode_column_safely(column_name, encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        logger.warning(f"Unseen value in {column_name}: {value}. Using default value 0")
        return 0  # Default to first class

# Ensure reasonable price predictions
def validate_price(predicted_price, min_price, max_price):
    if predicted_price < 0:
        logger.debug(f"Predicted price {predicted_price} is negative, adjusting to {max(0, min_price * 0.8)}")
        return max(0, min_price * 0.8)  # Floor at 0 or 80% of min_price
    if predicted_price > max_price * 2:
        logger.debug(f"Predicted price {predicted_price} exceeds max_price * 2, adjusting to {max_price * 1.2}")
        return max_price * 1.2  # Cap at 120% of max_price
    return predicted_price

@app.route('/')
def home():
    return "Welcome to the Crop Price Prediction API!"

@app.route('/predict', methods=['POST'])
@profile
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")

        # Load models and encoders
        mx_price, sc_price, xgb_model, nn_model = load_price_models()
        if not all([mx_price, sc_price, xgb_model, nn_model]):
            logger.error("One or more models/scalers failed to load")
            return jsonify({'error': 'Models not available'}), 500
        
        encoders = load_encoders()
        if not encoders:
            logger.error("No encoders loaded")
            return jsonify({'error': 'Encoders not available'}), 500

        # Encode categorical values
        encoded_data = {}
        for col in categorical_columns:
            if col in data:
                if col in encoders:
                    encoded_data[col] = encode_column_safely(col, encoders[col], data[col])
                else:
                    logger.error(f'Encoder for {col} not available')
                    return jsonify({'error': f'Encoder for {col} not available'}), 500
            else:
                logger.warning(f'Missing input for {col}, using default value 0')
                encoded_data[col] = 0
        
        # Add numerical values
        min_price = float(data.get('min_price', 0))
        max_price = float(data.get('max_price', 0))
        encoded_data['min_price'] = min_price
        encoded_data['max_price'] = max_price
        
        # Create input array
        input_array = np.array([[
            encoded_data.get('state', 0),
            encoded_data.get('district', 0),
            encoded_data.get('market', 0),
            encoded_data.get('crop_name', 0),
            encoded_data.get('min_price', 0),
            encoded_data.get('max_price', 0)
        ]])
        
        # Scale input data
        try:
            input_scaled = mx_price.transform(input_array)
            input_standardized = sc_price.transform(input_scaled)
        except Exception as e:
            logger.error(f"Error scaling input data: {str(e)}")
            return jsonify({'error': 'Failed to scale input data'}), 500
        
        # Make predictions
        try:
            xgb_pred = float(xgb_model.predict(input_standardized)[0])
            nn_pred = float(nn_model.predict(input_standardized, verbose=0)[0][0])
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Validate predictions
        xgb_pred = validate_price(xgb_pred, min_price, max_price)
        nn_pred = validate_price(nn_pred, min_price, max_price)
        
        # Calculate ensemble prediction
        ensemble_pred = (xgb_pred + nn_pred) / 2
        
        # Clean up memory
        gc.collect()
        
        # Return all predictions
        result = {
            'predicted_price_xgb': round(xgb_pred, 2),
            'predicted_price_nn': round(nn_pred, 2),
            'predicted_price_ensemble': round(ensemble_pred, 2),
            'min_input_price': min_price,
            'max_input_price': max_price
        }
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)  # Disable debug mode in production
