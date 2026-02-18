# Production Dockerfile for Phishing Detection Microservice
FROM python:3.10-slim-bookworm

# 1. Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# 2. Install system dependencies for whois/dns and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    whois \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code and model registry
COPY . .

# 5. Expose port
EXPOSE 8000

# 6. Production entrypoint
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
