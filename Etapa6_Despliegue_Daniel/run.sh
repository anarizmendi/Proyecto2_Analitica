#!/bin/bash

# Salir si algo falla
set -e

# Ir a la carpeta del proyecto (un nivel arriba de este archivo)
cd "$(dirname "$0")/.."

# Nombre de la imagen y contenedor
IMAGE_NAME="airbnb-tokyo:local"
CONTAINER_NAME="airbnb-tokyo"

echo "Deteniendo contenedor anterior (si existe)..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "Construyendo imagen..."
docker build -t "$IMAGE_NAME" .

echo "Levantando contenedor..."
docker run -d --name "$CONTAINER_NAME" -p 8050:8050 "$IMAGE_NAME"

echo "Listo. La app debería estar disponible en http://localhost:8050"


