# Base officielle PyTorch avec CUDA (optimisée pour l'IA)
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Éviter les questions interactives lors de l'installation de paquets Linux
ENV DEBIAN_FRONTEND=noninteractive

# Installation des paquets système (ffmpeg est obligatoire pour manipuler l'audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Définition du dossier de travail dans le conteneur
WORKDIR /app

# Copie des pré-requis en premier (optimisation du cache Docker)
COPY requirements.txt .

# Mise à jour de pip et installation des bibliothèques Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du script principal
COPY handler.py .

# Commande par défaut au démarrage du conteneur
CMD ["python", "-u", "handler.py"]