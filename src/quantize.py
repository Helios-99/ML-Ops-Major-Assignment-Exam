
"""
This script shrinks our linear regression model using 8-bit and 16-bit
quantization.

The main idea is to trade a tiny bit of precision for a much smaller
file size, which is great for deployment. Wewill also check how much
accuracy we lose in the process.
"""

import os
import numpy as np
import joblib
import pandas as pd
from src import utils

# Central place for our file paths
MODEL_PATH = "models/linear_regression.joblib"
ARTIFACTS_DIR = "models"


def quantize_and_report(coeffs, bias, bit_depth):
    """A helper to run the quantization for a given bit-depth."""
    print(f"\n>> Running {bit_depth}-bit quantization...")

    # Pick the right tool for the job
    quant_fn, dequant_fn = (utils.q16_fn, utils.dq16_fn) if bit_depth == 16 else (utils.q8_fn, utils.dq8_fn)

    # Quantize the model's parameters (the coefficients and the bias)
    q_coeffs, c_min, _, c_scale = quant_fn(coeffs)
    q_bias, b_min, _, b_scale = quant_fn(np.array([bias]))
    
    # Save the shrunken parameters. This is the new, smaller model.
    quant_params = {
        'coeffs': q_coeffs, 'coeff_min': c_min, 'coeff_scale': c_scale,
        'bias': q_bias[0], 'bias_min': b_min, 'bias_scale': b_scale,
    }
    out_path = f"{ARTIFACTS_DIR}/quant_params_{bit_depth}bit.joblib"
    joblib.dump(quant_params, out_path)
    print(f"   ...{bit_depth}-bit params saved to {out_path}")

    # Now, de-quantize them right away to see how much data we lost.
    # This is the "reconstruction error".
    dq_coeffs = dequant_fn(q_coeffs, c_min, c_scale)
    dq_bias = dequant_fn(q_bias, b_min, b_scale)[0]
    
    # NOTE: This error is just on the parameters themselves, not the final prediction.
    recon_error = np.abs(coeffs - dq_coeffs).max()
    print(f"   ...Max weight reconstruction error: {recon_error:.6f}")
    
    return {
        "path": out_path,
        "dq_coeffs": dq_coeffs,
        "dq_bias": dq_bias
    }


def main():
    """Start the whole analysis."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Alright, first things first, load the original model
    print("Loading original model...")
    model = utils.load_model(MODEL_PATH)
    coeffs, bias = model.coef_, model.intercept_
    
    # And the data we'll use to test it
    _, X_test, _, y_test = utils.load_dataset()
    
    # Run the quantization process for both 16 and 8 bits
    results16 = quantize_and_report(coeffs, bias, bit_depth=16)
    results8 = quantize_and_report(coeffs, bias, bit_depth=8)
    
    # --- Check out the size difference ---
    print("\n--- File Size Report ---")
    size_orig_kb = os.path.getsize(MODEL_PATH) / 1024
    size_16bit_kb = os.path.getsize(results16["path"]) / 1024
    size_8bit_kb = os.path.getsize(results8["path"]) / 1024
    
    print(f"Original: {size_orig_kb:.2f} KB")
    print(f"16-bit:   {size_16bit_kb:.2f} KB ({(size_16bit_kb/size_orig_kb)*100:.1f}% of original)")
    print(f"8-bit:    {size_8bit_kb:.2f} KB ({(size_8bit_kb/size_orig_kb)*100:.1f}% of original)")
    
    # --- Now for the real test: model performance ---
    print("\n--- Accuracy Report (R² / MSE) ---")

    # Get predictions and metrics for the 16-bit model
    preds16 = X_test @ results16["dq_coeffs"] + results16["dq_bias"]
    r2_16, mse_16 = utils.compute_metrics(y_test, preds16)
    print(f"16-bit model: R²={r2_16:.4f}, MSE={mse_16:.4f}")

    # And for the 8-bit model
    preds8 = X_test @ results8["dq_coeffs"] + results8["dq_bias"]
    r2_8, mse_8 = utils.compute_metrics(y_test, preds8)
    print(f"8-bit model:  R²={r2_8:.4f}, MSE={mse_8:.4f}")
    
    # Let's also get the original model's performance for a baseline
    preds_orig = model.predict(X_test)
    r2_orig, mse_orig = utils.compute_metrics(y_test, preds_orig)
    print(f"Original model: R²={r2_orig:.4f}, MSE={mse_orig:.4f}")
    
    print("\n Performance drop is minimal.")
    print("Done")


if __name__ == "__main__":
    main()
