FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

ENV PORT 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

# --- Build and Run Locally (for testing Dockerfile) ---
# Ensure image_classifier.onnx exists before building.
# 1. Build: docker build -t image-classifier-app .
# 2. Run:   docker run -p 8000:80 image-classifier-app
#    Then access via http://localhost:8000/docs or send POST to http://localhost:8000/predict