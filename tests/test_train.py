# tests/test_pipeline.py

import pytest
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from src import utils

MINIMUM_R2_SCORE = 0.5

@pytest.fixture(scope='module')
def dataset():
    """
    Fixture to load the dataset once for all tests in this file.
    Saves time by not reloading the data for every single test.
    """
    print("\n(Setting up dataset fixture...)")
    X_train, X_test, y_train, y_test = utils.load_dataset()
    return X_train, X_test, y_train, y_test

def test_data_shapes_and_split(dataset):
    """Checks if the dataset dimensions and split ratio are correct."""
    X_train, X_test, y_train, y_test = dataset
    
    assert X_train.shape[0] == y_train.shape[0], "Train sets length mismatch"
    assert X_test.shape[0] == y_test.shape[0], "Test sets length mismatch"
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch"
    
    # Check if the split ratio is roughly 80/20
    train_ratio = len(X_train) / (len(X_train) + len(X_test))
    assert 0.78 <= train_ratio <= 0.82, "Train/test split ratio is not ~80/20"

def test_model_training(dataset):
    """
    Tests if the model gets trained and develops the necessary attributes
    (coefficients and an intercept).
    """
    X_train, _, y_train, _ = dataset
    model = LinearRegression()
    
    with pytest.raises(AttributeError):
        _ = model.coef_
        
    model.fit(X_train, y_train)
    assert hasattr(model, 'coef_'), "Model should have 'coef_' after training."
    assert hasattr(model, 'intercept_'), "Model should have 'intercept_' after training."
    assert model.coef_ is not None

def test_model_performance(dataset):
    """
    Checks if the trained model's performance on the test set is above
    our minimum acceptable threshold.
    """
    X_train, X_test, y_train, y_test = dataset
    
    # Train a model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    r2, _ = utils.calculate_regression_performance(y_test, predictions)
    
    assert r2 > MINIMUM_R2_SCORE, f"Model R2 score {r2:.3f} was below the threshold of {MINIMUM_R2_SCORE}"

def test_model_persistence(dataset, tmp_path):
    """
    Tests if the model can be saved and loaded without changing.
    """
    X_train, X_test, y_train, _ = dataset
    model = LinearRegression()
    model.fit(X_train, y_train)

    model_path = tmp_path / "test_model.joblib"

    utils.save_artifact(model, model_path)
    loaded_model = utils.load_artifact(model_path)

    assert os.path.exists(model_path)
    original_predictions = model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)
    np.testing.assert_allclose(original_predictions, loaded_predictions)
