# Use a Python 3.10 slim image for a smaller footprint
FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY src2/requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --upgrade pip setuptools wheel \
    && pip install tensorflow==2.16.1 \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && pip install --no-cache-dir flask

# Copy all application code from src2 into the container's /app directory
COPY src/ /app/

# Copy the model artifacts from your specified subdirectory
COPY src/model/ /app/

# Add a 'serve' executable for AWS SageMaker
RUN echo '#!/bin/sh\npython serve.py' > /usr/local/bin/serve \
    && chmod +x /usr/local/bin/serve

# Expose the port specified in your serve.py
EXPOSE 8080

# Command to run your application
CMD ["python", "serve.py"]
