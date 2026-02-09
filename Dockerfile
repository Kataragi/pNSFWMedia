FROM python:3.11-slim

# System dependencies (git is needed for CLIP install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces expects user with uid 1000
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Install Python build tools (pin setuptools<70 to keep pkg_resources)
RUN pip install --no-cache-dir --upgrade pip "setuptools<70" wheel

# Install PyTorch CPU-only (~200MB instead of ~2GB)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install CLIP from GitHub
# --no-build-isolation: use the pinned setuptools (not latest)
# --no-deps: avoid re-pulling torch
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    git+https://github.com/openai/CLIP.git

# Install remaining requirements
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy ALL application code
COPY . .

# Fix ownership for the non-root user
RUN chown -R user:user /home/user/app

# Switch to non-root user
USER user

# HuggingFace Spaces configuration
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
