FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio and Python packages
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    libespeak-dev \
    libpulse-dev \
    alsa-utils \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories if they don't exist
RUN mkdir -p models temp_audio

# Command to run the application
CMD ["python", "main.py"]