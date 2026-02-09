FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces expects user with uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install Python build tools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU-only (~200MB instead of ~2GB)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install CLIP from GitHub (--no-deps to skip re-resolving torch)
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/openai/CLIP.git

# Install remaining requirements
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy application code
COPY --chown=user . .

# Switch to non-root user
USER user

# HuggingFace Spaces configuration
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
