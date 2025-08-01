import numpy as np
from sklearn.linear_model import LinearRegression
from src import utils

MODEL_OUTPUT_PATH = "models/linear_regression.joblib"

def execute_training_cycle():
    print("Loading and preparing dataset...")
    x_train, x_test, y_train, y_test = utils.load_dataset()

    print("Initializing regression model...")
    regressor = LinearRegression()

    print("Training model on the dataset...")
    regressor.fit(x_train, y_train)

    print("Evaluating model performance...")
    predictions = regressor.predict(x_test)
    
    r2_val, mse_val = utils.calculate_regression_performance(y_test, predictions)
    absolute_errors = np.abs(y_test - predictions)

    print("\n--- Training Run Report ---")
    print(f"  R-squared: {r2_val:.4f}")
    print(f"  Mean Squared Error: {mse_val:.4f}")
    print(f"  Mean Absolute Error: {absolute_errors.mean():.4f}")
    print(f"  Max Prediction Error: {absolute_errors.max():.4f}")
    print("---------------------------\n")

    utils.save_artifact(regressor, MODEL_OUTPUT_PATH)
    print(f"Model successfully saved to: {MODEL_OUTPUT_PATH}")

    return regressor

if __name__ == "__main__":
    execute_training_cycle()
