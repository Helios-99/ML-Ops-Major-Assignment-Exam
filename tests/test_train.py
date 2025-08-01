# tests/test_pipeline.py

import pytest
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# We assume the project is installed in editable mode (`pip install -e .`)
# so we can import directly from 'src' without changing sys.path.
from src import utils

# A reasonable R-squared score to expect from the model.
# We set it as a constant so its easy to find and change.
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
    
    # Before training, these attributes shouldn't exist.
    with pytest.raises(AttributeError):
        _ = model.coef_
        
    # After training, they must exist.
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
    
    # Make predictions and check performance
    predictions = model.predict(X_test)
    r2, _ = utils.compute_metrics(y_test, predictions)
    
    assert r2 > MINIMUM_R2_SCORE, f"Model R2 score {r2:.3f} was below the threshold of {MINIMUM_R2_SCORE}"


def test_model_persistence(dataset, tmp_path):
    """
    Can we save a trained model and load it back correctly?
    This test ensures the save/load cycle doesn't corrupt the model.
    
    Uses pytest's built-in `tmp_path` fixture for clean, temporary file handling.
    """
    X_train, X_test, y_train, _ = dataset
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Define a temporary path for our test model
    model_path = tmp_path / "test_model.joblib"
    
    # Save and then load the model
    utils.save_model(model, model_path)
    assert os.path.exists(model_path), "Model file was not created."
    
    loaded_model = utils.load_model(model_path)
    
    # The loaded model should make identical predictions to the original.
    original_preds = model.predict(X_test)
    loaded_preds = loaded_model.predict(X_test)
    
    np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-6)
