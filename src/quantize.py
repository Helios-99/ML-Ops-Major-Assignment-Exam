import os
import numpy as np
import pandas as pd
from src import utils

MODEL_PATH = "models/linear_regression.joblib"
ARTIFACTS_DIR = "models"


def run_quantization_and_evaluation():
    """
    Main workflow to load a model, quantize it to 8 and 16 bits,
    and report on the size reduction and performance impact.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    print("Loading original trained model...")
    original_model = utils.load_artifact(MODEL_PATH)
    coefficients = original_model.coef_
    intercept = original_model.intercept_

    print("Loading test data for evaluation...")
    _, X_test, _, y_test = utils.prepare_california_housing_data()

    # This dictionary will hold the results for each bit depth
    quantization_results = {}

    for n_bits in [16, 8]:
        print(f"\n--- Processing {n_bits}-bit Quantization ---")
        
        q_coeffs, c_min, c_range = utils.convert_to_quantized(coefficients, num_bits=n_bits)
        q_intercept, i_min, i_range = utils.convert_to_quantized(np.array([intercept]), num_bits=n_bits)
        
        artifact_path = f"{ARTIFACTS_DIR}/quant_params_{n_bits}bit.joblib"
        utils.save_artifact({
            'q_coeffs': q_coeffs,
            'coeff_min': c_min,
            'coeff_range': c_range,
            'q_intercept': q_intercept,
            'intercept_min': i_min,
            'intercept_range': i_range
        }, artifact_path)
        print(f"Saved {n_bits}-bit artifacts to {artifact_path}")
        
        dq_coeffs = utils.revert_from_quantized(q_coeffs, c_min, c_range)
        dq_intercept = utils.revert_from_quantized(q_intercept, i_min, i_range)[0]
        
        quantization_results[n_bits] = {
            "path": artifact_path,
            "dq_coeffs": dq_coeffs,
            "dq_intercept": dq_intercept
        }

    #
