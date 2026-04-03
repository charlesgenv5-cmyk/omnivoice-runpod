# Base officielle PyTorch avec CUDA
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Installation des outils systèmes
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# On installe les librairies Python ET huggingface_hub
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    huggingface_hub

# LE FIX EST ICI 🚨 : On télécharge le modèle DANS l'image directement sur GitHub
RUN huggingface-cli download k2-fsa/OmniVoice

COPY handler.py .

CMD ["python", "-u", "handler.py"]
