import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def load_dataset(split_ratio=0.2, seed=42):
    dataset = fetch_california_housing()
    features_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    target_series = pd.Series(dataset.target, name="MedianValue")
    
    x_train, x_test, y_train, y_test = train_test_split(
        features_df, target_series, test_size=split_ratio, random_state=seed
    )
    return x_train, x_test, y_train, y_test

def save_artifact(obj, file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    joblib.dump(obj, file_path)

def load_artifact(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact not found at: {file_path}")
    return joblib.load(file_path)

def calculate_regression_performance(y_true, y_predicted):
    score_r2 = r2_score(y_true, y_predicted)
    error_mse = mean_squared_error(y_true, y_predicted)
    return score_r2, error_mse

def _get_quantization_params(num_bits):
    if num_bits == 8:
        return np.uint8, 255.0
    elif num_bits == 16:
        return np.uint16, 65535.0
    else:
        raise ValueError("Bit depth must be 8 or 16.")

def convert_to_quantized(float_array, num_bits):
    int_type, max_int_val = _get_quantization_params(num_bits)
    
    min_val = float_array.min()
    max_val = float_array.max()
    value_range = max_val - min_val

    if value_range == 0:
        quantized = np.full(float_array.shape, int(max_int_val / 2), dtype=int_type)
        return quantized, min_val, value_range

    quantized = (((float_array - min_val) / value_range) * max_int_val).round().astype(int_type)
    return quantized, min_val, value_range

def revert_from_quantized(quantized_array, min_val, value_range):
    _, max_int_val = _get_quantization_params(quantized_array.dtype.itemsize * 8)

    if value_range == 0:
        return np.full(quantized_array.shape, min_val, dtype=np.float32)

    float_array = (quantized_array.astype(np.float32) / max_int_val) * value_range + min_val
    return float_array
