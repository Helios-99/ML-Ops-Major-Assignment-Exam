# Use Python 3.9 slim image
FROM python:3.9-slim

# Set the main working directory
WORKDIR /app

# Copy requirements for dependency caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and model files
COPY src/ ./src/
COPY models/ ./models/

# Set PYTHONPATH for module resolution
ENV PYTHONPATH=/app/src

# Default entrypoint for container
CMD ["python", "-m", "src.predict"]
