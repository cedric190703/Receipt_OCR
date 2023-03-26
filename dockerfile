# Utiliser une image de Python 3.9 comme base
FROM python:3.9-slim-buster

# Installer les dépendances système nécessaires pour Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1\
    && rm -rf /var/lib/apt/lists/*

# Créer et travailler dans un répertoire de travail pour l'application
WORKDIR /app

# Copier tout le contenu du répertoire courant dans le répertoire de travail
COPY . /app

# Installer les dépendances Python de l'application dans requirements.txt
RUN pip install -r requirements.txt

# Exposer le port 5000 pour la communication avec l'application
EXPOSE 5000

# Démarrer l'application Flask
ENV FLASK_APP=apiApp.py
CMD ["flask", "run", "--host=0.0.0.0"]
