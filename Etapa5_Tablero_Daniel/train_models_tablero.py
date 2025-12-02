"""
Script para entrenar modelos sencillos de regresión y clasificación
para el tablero de Daniel y guardarlos en la carpeta /models.

Se asume que existe modelo/listings_clean.csv
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


# -------------------------------------------------------------------
# 1. Cargar datos limpios
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # carpeta raíz del repo
data_path = os.path.join(BASE_DIR, "modelo", "listings_clean.csv")

print(f"📂 Cargando datos desde: {data_path}")
df = pd.read_csv(data_path)

print(f"✅ Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")

# -------------------------------------------------------------------
# 2. Definir variables objetivo y features
# -------------------------------------------------------------------
# Target de regresión: price
if "price" not in df.columns:
    raise ValueError("La columna 'price' no existe en listings_clean.csv")

df = df.dropna(subset=["price"])

# Target de clasificación: 1 si el precio está por encima de la mediana, 0 si no
median_price = df["price"].median()
df["high_price"] = (df["price"] > median_price).astype(int)

print(f"💰 Mediana de precio: {median_price:.2f}")

# Features candidatas (ajusta si alguna no existe en tu CSV)
candidate_numeric = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "accommodates",
]

candidate_categorical = [
    "room_type",
    "neighbourhood_group",
    "neighbourhood",
]

numeric_features = [c for c in candidate_numeric if c in df.columns]
categorical_features = [c for c in candidate_categorical if c in df.columns]

if not numeric_features and not categorical_features:
    raise ValueError("No se encontraron columnas numéricas ni categóricas válidas.")

print("🔢 Features numéricas usadas:", numeric_features)
print("🔤 Features categóricas usadas:", categorical_features)

feature_cols = numeric_features + categorical_features
X = df[feature_cols]
y_reg = df["price"]
y_clf = df["high_price"]

# -------------------------------------------------------------------
# 3. Partición train/test
# -------------------------------------------------------------------
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print(f"📊 Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

# -------------------------------------------------------------------
# 4. Definir preprocesador y modelos
# -------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Pipeline de REGRESIÓN
reg_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )),
    ]
)

# Pipeline de CLASIFICACIÓN
clf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )),
    ]
)

# -------------------------------------------------------------------
# 5. Entrenar modelos
# -------------------------------------------------------------------
print("\n🚀 Entrenando modelo de REGRESIÓN...")
reg_pipeline.fit(X_train, y_reg_train)

y_reg_pred = reg_pipeline.predict(X_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = mse ** 0.5  # raíz cuadrada del MSE
r2 = r2_score(y_reg_test, y_reg_pred)
print(f"   ➜ RMSE: {rmse:.2f}")
print(f"   ➜ R2:   {r2:.3f}")


print("\n🚀 Entrenando modelo de CLASIFICACIÓN...")
clf_pipeline.fit(X_train, y_clf_train)

y_clf_pred = clf_pipeline.predict(X_test)
acc = accuracy_score(y_clf_test, y_clf_pred)
print(f"   ➜ Accuracy: {acc:.3f}")

# -------------------------------------------------------------------
# 6. Guardar modelos en /models
# -------------------------------------------------------------------
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

reg_path = os.path.join(models_dir, "regression_pipeline_tablero.joblib")
clf_path = os.path.join(models_dir, "classification_pipeline_tablero.joblib")

joblib.dump(reg_pipeline, reg_path)
joblib.dump(clf_pipeline, clf_path)

print("\n💾 Modelos guardados:")
print(f"   - {reg_path}")
print(f"   - {clf_path}")
print("\n✅ Entrenamiento y guardado de modelos COMPLETADO.")
