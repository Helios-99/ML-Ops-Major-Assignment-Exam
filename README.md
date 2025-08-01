# MLOps Pipeline for Linear Regression
## Description
This was my major assignment for an MLOps course. The goal was to build a complete, automated pipeline for a basic scikit-learn Linear Regression model. It handles everything from running tests to training the model, quantizing it, and finally packaging it into a Docker container.

## Key Features
Automated CI/CD: The entire pipeline is triggered on every push to the main branch.

Model Training & Evaluation: Trains a model and evaluates its performance.

Unit Testing: A pytest suite ensures the reliability of the data processing and training code.

Model Quantization: A custom implementation of 8-bit and 16-bit quantization.

Docker Containerization: The prediction script is packaged into a Docker image for portability.

## Comparison Table
Quantization reduces model size at the cost of some precision. The table below shows that 16-bit quantization provides a good balance, significantly reducing size with almost no impact on accuracy. In contrast, 8-bit quantization severely degrades performance for this model.
| Model Version | R² Score | MSE | Size (KB) |
| :--- | :---: | :---: | :---: |
| Original (Float64) | 0.576 | 0.556 | ~0.65 KB |
| Quantized (16-bit) | 0.575 | 0.557 | ~0.55 KB |
| Quantized (8-bit) | -46.683| 62.484| ~0.53 KB |

### Getting Started
Prerequisites: You'll need Python 3.9+, Git, and Docker installed.

To get a local copy up and running:

Bash

### 1. Clone the repository

### 2. Set up a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Use `.\venv\Scripts\Activate.ps1` on PowerShell
pip install -r requirements.txt
How to Use
You can run each part of the process individually.

Bash

### Run the test suite
pytest

### Train the model
python -m src.train

### Run quantization analysis
python -m src.quantize
Docker Usage
To build and run the application as a Docker container:

Bash

### 1. Build the image
docker build -t regression-app .

### 2. Run the container
docker run --rm regression-app
CI/CD Pipeline
The workflow is defined in .github/workflows/ci.yml and automates the entire process in three stages:

validate_codebase: Runs all pytest tests.

build_model_artifacts: Trains and quantizes the model, then saves the model files.

package_and_verify_image: Builds the Docker image with the saved models and runs it as a final test.

## Conclusion
16-bit quantization is the clear winner here. It provides a decent size reduction with almost no drop in performance. 8-bit quantization is too aggressive for this model and completely breaks its predictive power (a negative R² score is a very bad sign).
