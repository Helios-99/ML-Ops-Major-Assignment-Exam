# src/predict.py

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from src import utils

# Define model path as a constant
MODEL_PATH = "models/linear_regression.joblib"

def validate_model_performance():
    """
    Loads the trained model, runs inference on the test set,
    and prints evaluation metrics and sample predictions.
    """
    print("--- Starting Model Validation Process ---")
    
    try:
        # 1. Load the trained model artifact
        regressor = utils.load_artifact(MODEL_PATH)
        print(f"Model '{MODEL_PATH}' loaded successfully.")

        # 2. Load the test dataset
        _, X_test, _, y_test = utils.load_dataset()
        print("Test data loaded.")

    except FileNotFoundError as e:
        print(f"ERROR: Could not load assets. {e}")
        print("Please ensure you have run the training script first.")
        return

    # 3. Generate predictions on the test data
    print("Running inference on the test set...")
    predictions = regressor.predict(X_test)

    # 4. Calculate performance metrics
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print("\n--- Model Performance on Test Set ---")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print("-------------------------------------\n")

    # 5. Display a few sample predictions for qualitative review
    print("Showing first 5 sample predictions:")
    results = pd.DataFrame({
        'Actual': y_test.head(5).values,
        'Predicted': predictions[:5]
    })
    results['Difference'] = (results['Actual'] - results['Predicted']).abs()
    print(results.to_string(index=False, float_format="%.2f"))

    print("\n Validation process completed successfully.")


if __name__ == "__main__":
    validate_model_performance()
