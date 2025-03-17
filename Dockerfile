FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files for FastSAM
COPY fastsam/ /app/fastsam/
COPY ultralytics/ /app/ultralytics/
COPY weights/ /app/weights/
COPY handler.py /app/handler.py

# Set up environment
ENV PYTHONPATH=/app

# Default command - will be overwritten by Hugging Face Inference Endpoints
CMD ["python", "-c", "import handler; print('Model loaded successfully')"]