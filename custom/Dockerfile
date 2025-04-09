FROM python:3.9-slim

WORKDIR /app/custom

# Install git for any potential git dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy your code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir requests

# Command to run your application
CMD ["python3", "main.py"]