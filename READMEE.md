# 📄 **README.md — Proyecto 2: Analítica Airbnb Tokyo**

## 🏙️ **Contexto del Problema**

Este proyecto analiza el mercado de alojamientos de **Airbnb en Tokio**, con el objetivo de:

1. **Predecir precios por noche** usando modelos de *regresión*.
2. **Clasificar propiedades como “recomendadas” o “no recomendadas”**, con base en sus características (*clasificación*).
3. Integrar los modelos en un **tablero interactivo** desarrollado en Dash y desplegado en AWS mediante contenedores Docker.

El proyecto sigue la metodología y etapas solicitadas en el curso *Analítica Computacional para la Toma de Decisiones* (Uniandes).

---

## 📁 **Estructura del Repositorio**

### ### **1️⃣ Carpeta `/modelo/` — Modelos Predictivos y Preprocesamiento**

Esta carpeta contiene **toda la lógica de modelado**, incluyendo:

* `Modelos_Predictivos_Andes.ipynb`
  Notebook principal donde se realizaron:

  * Limpieza profunda de datos
  * Ingeniería de características
  * Imputación y codificación
  * Preprocesamiento con *sklearn pipelines*
  * Modelos baseline (Regresión Lineal, Ridge, Random Forest)
  * Redes neuronales para regresión y clasificación
  * Búsqueda amplia de hiperparámetros
  * Selección del mejor modelo basado en métricas
  * Serialización de modelos y preprocesadores

* `best_regression_model.keras`
  Modelo final de regresión, entrenado y guardado para despliegue.

* `best_classification_model.keras`
  Modelo final de clasificación.

* `preprocessor_regression.joblib`
  Pipeline de preprocesamiento usado para el modelo de regresión.

* `preprocessor_classification.joblib`
  Pipeline para el modelo de clasificación.

Estos objetos permiten que el tablero cargue los modelos **pre-entrenados**, sin necesidad de reentrenar.

---

### ### **2️⃣ Seguimiento de Experimentos con MLflow**

Durante el desarrollo se registraron múltiples configuraciones de modelos usando **MLflow**, lo cual permite:

* comparar arquitecturas,
* evaluar hiperparámetros,
* visualizar métricas de entrenamiento,
* revisar curvas de pérdida,
* y almacenar modelos entrenados.

Para visualizar los experimentos localmente, ejecutar:

```bash
mlflow ui
```

Luego abrir en el navegador:

```
http://127.0.0.1:5000
```

---

## 📁 **3️⃣ Carpeta `/despliegue/` — Tablero en Dash + Docker + AWS**

Esta carpeta contiene el trabajo realizado por el integrante encargado del despliegue.
Incluye:

### **🔧 Etapa 5 — Diseño y Desarrollo del Tablero**

* Código en Dash (`app.py`)
* Wireframe / mockup del tablero
* Archivos `assets/` (CSS, estilos)
* Visualizaciones solicitadas por el proyecto
* Integración de los modelos serializados para:

  * predecir precio por noche
  * clasificar un nuevo listing

El tablero permite que el usuario:

* Ingrese valores de un listing
* Obtenga predicción del precio
* Obtenga una recomendación mediante el modelo de clasificación
* Visualice métricas y gráficas relevantes del mercado de Tokio

### **☁️ Etapa 6 — Despliegue en AWS con Docker**

Incluye:

* `Dockerfile`
* `requirements.txt`
* Instrucciones para levantar el contenedor
* Archivos necesarios para ejecutar el tablero en EC2
* Screenshots del despliegue en AWS

  * IP pública
  * IP privada
  * usuario EC2
  * contenedor corriendo y accesible desde internet

El tablero final debe quedar disponible en una URL pública.


---

## **Instrucciones Generales para Correr el Proyecto Localmente**

### 1. Crear entorno virtual

```bash
python3 -m venv airbnb-env
source airbnb-env/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el notebook

Abrir `modelo/Modelos_Predictivos_Andes.ipynb`.

### 4. Ejecutar MLflow (opcional)

```bash
mlflow ui
```

### 5. Cargar y ejecutar el tablero

```bash
cd despliegue
python app.py
```

---

## 👥 **Autores**

* **Ana Sofía Arizmendi** – Modelos predictivos (regresión y clasificación)
* **Daniel Mitchell** – Diseño del tablero, despliegue en AWS y Docker

