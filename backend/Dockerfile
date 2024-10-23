# Backend Dockerfile
FROM python:3.10.12-slim

RUN apt-get update && \
apt-get install -y --no-install-recommends \
libopencv-dev && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app /app

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
