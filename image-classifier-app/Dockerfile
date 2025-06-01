# Base image
FROM python:3.10-slim
RUN apt-get update && apt-get install dumb-init
RUN update-ca-certificates
WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# Source code
COPY . .

# Configuration
EXPOSE 8192
CMD ["dumb-init", "--", "fastapi", "run", "app.py", "--port", "8192"]