# Lightweight Python image
FROM python:3.11-slim

# Install system dependencies (none heavy required here)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command runs the demo pipeline
CMD ["python", "app/main.py", "--epochs", "50", "--lr", "0.1"]
