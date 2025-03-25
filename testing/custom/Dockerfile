FROM python:3.9-slim

WORKDIR /app/custom

# Copy your code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir requests

# Command to run your application
CMD ["python3", "main.py"]