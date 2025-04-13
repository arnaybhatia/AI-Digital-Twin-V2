FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv

# Install necessary system packages including git
RUN apt update && \
    apt install -y espeak-ng git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/zonos

# Clone the Zonos repository
RUN git clone https://github.com/Zyphra/Zonos.git .

# Install Zonos using uv for faster installation
RUN uv pip install --system -e . && uv pip install --system -e .[compile]

# Create the generate script
COPY zonos_generate.py /app/zonos/

# Create a simple script that keeps the container running
RUN echo 'import time\n\
import os\n\
import signal\n\
import sys\n\
\n\
# Function to handle signals properly\n\
def signal_handler(sig, frame):\n\
    print("\\nShutting down gracefully...")\n\
    sys.exit(0)\n\
\n\
# Register signal handlers\n\
signal.signal(signal.SIGTERM, signal_handler)\n\
signal.signal(signal.SIGINT, signal_handler)\n\
\n\
print("Zonos service is ready and waiting for commands.")\n\
print("Use docker exec to run text-to-speech commands.")\n\
\n\
# Keep container running\n\
while True:\n\
    try:\n\
        time.sleep(60)  # Just sleep and keep container running\n\
    except KeyboardInterrupt:\n\
        print("\\nShutting down...")\n\
        break\n\
' > server.py

# Set the entrypoint to Python
ENTRYPOINT ["python"]

# Command to run the server
CMD ["server.py"]