# Imagen base de Python (ligera)
FROM python:3.12-slim

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Copiamos TODO el proyecto dentro del contenedor
COPY . .

# Instalamos las dependencias necesarias para el tablero
RUN pip install --no-cache-dir \
    dash==3.2.0 \
    dash-bootstrap-components==1.6.0 \
    pandas==2.3.1 \
    scikit-learn==1.7.2 \
    joblib==1.5.2 \
    plotly==6.3.0 \
    numpy==2.3.2

# Puerto donde corre el tablero dentro del contenedor
EXPOSE 8050

# Comando que se ejecuta cuando se levanta el contenedor
CMD ["bash", "Etapa6_Despliegue_Daniel/run.sh"]

