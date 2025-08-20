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
# Copy the necessary scripts
COPY zonos_generate.py server.py /app/zonos/
# Set the entrypoint and command for the container runtime
ENTRYPOINT ["python"]
CMD ["server.py"]