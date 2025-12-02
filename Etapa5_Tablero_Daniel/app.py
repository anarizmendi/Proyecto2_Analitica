# Etapa5_Tablero_Daniel/app.py
# Tablero Airbnb Tokyo - Daniel (Plan B: modelos sklearn)

import os
import pandas as pd
import joblib
import numpy as np

from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px

# ---------------------------
# RUTAS Y CARGA DE DATOS
# ---------------------------

# Carpeta raíz del repo (Proyecto2_Analitica)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(ROOT_DIR, "modelo", "listings_clean.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

print("===== CARGANDO TABLERO DASH (Plan B - sklearn) =====")
print(f"✔ Cargando datos desde: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"✔ Dataset cargado con {len(df)} filas y {df.shape[1]} columnas")

print(f"✔ Cargando modelos desde: {MODELS_DIR}")
reg_pipeline = joblib.load(os.path.join(MODELS_DIR, "regression_pipeline_tablero.joblib"))
clf_pipeline = joblib.load(os.path.join(MODELS_DIR, "classification_pipeline_tablero.joblib"))
print("✔ Modelos sklearn cargados")

# ---------------------------
# FIGURAS EDA
# ---------------------------

# Histograma de precios
fig_hist = px.histogram(
    df,
    x="price",
    nbins=60,
    title="Distribución de precios",
)
fig_hist.update_layout(
    height=250,
    margin=dict(l=10, r=10, t=40, b=30),
)

# Heatmap: precio promedio según reseñas y tipo de habitación

bins = [0, 10, 25, 50, 100, 200, df["number_of_reviews"].max()]
labels = ["0–10", "11–25", "26–50", "51–100", "101–200", "200+"]

df["reviews_bin"] = pd.cut(
    df["number_of_reviews"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

heat_data = (
    df.groupby(["reviews_bin", "room_type"])["price"]
    .mean()
    .reset_index()
)

heat_pivot = heat_data.pivot(
    index="reviews_bin",
    columns="room_type",
    values="price",
)

fig_heatmap = px.imshow(
    heat_pivot,
    aspect="auto",
    labels=dict(
        x="Tipo de habitación",
        y="Rango de reseñas totales",
        color="Precio medio (JPY)",
    ),
    title="Precio promedio según reseñas y tipo de habitación",
)
fig_heatmap.update_layout(
    height=320,
    margin=dict(l=20, r=20, t=40, b=40),
)

# Scatter dinámico: precio vs capacidad (filtrado por noches mínimas)
def make_scatter(min_nights):
    df_f = df[df["minimum_nights"] >= min_nights]
    fig = px.scatter(
        df_f,
        x="accommodates",
        y="price",
        color="room_type",
        labels={"accommodates": "Capacidad (huéspedes)", "price": "Precio (JPY)"},
        title="Precio vs capacidad",
        height=380,
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=50))
    return fig

# KPIs generales
num_listings = len(df)
avg_price = df["price"].mean()
top_rooms = df["room_type"].value_counts().head(3)

# ---------------------------
# APP DASH
# ---------------------------

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": "#E0F7FF", "padding": "16px"},
    children=[
        # TÍTULO
        html.Div(
            [
                html.H2("Conoce el precio de tu AirBnB", style={"fontWeight": "700"}),
                html.P(
                    "Este tablero permite predecir el precio de tu alojamiento en Tokio a partir de variables clave.",
                    style={"marginBottom": "0"},
                ),
            ],
            style={"marginBottom": "16px"},
        ),

        dbc.Row(
            [
                # --------------------------------------
                # COLUMNA IZQUIERDA (inputs + modelo + scatter)
                # --------------------------------------
                dbc.Col(
                    width=9,
                    children=[
                        # Panel de inputs
                        html.Div(
                            style={
                                "backgroundColor": "#CCF2FF",
                                "padding": "12px 16px",
                                "borderRadius": "8px",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.H5("Ingresa los datos de tu propiedad"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Tipo de habitación"),
                                                dcc.Dropdown(
                                                    id="input-room-type",
                                                    options=[
                                                        {"label": rt, "value": rt}
                                                        for rt in df["room_type"]
                                                        .dropna()
                                                        .unique()
                                                    ],
                                                    value=df["room_type"].mode()[0],
                                                    clearable=False,
                                                ),
                                            ],
                                            md=3,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Huéspedes"),
                                                dcc.Input(
                                                    id="input-accommodates",
                                                    type="number",
                                                    min=1,
                                                    max=16,
                                                    value=2,
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Disponibilidad anual (días)"
                                                ),
                                                dcc.Input(
                                                    id="input-availability",
                                                    type="number",
                                                    min=0,
                                                    max=365,
                                                    value=120,
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Reseñas totales"),
                                                dcc.Input(
                                                    id="input-num-reviews",
                                                    type="number",
                                                    min=0,
                                                    value=10,
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Reseñas / mes"),
                                                dcc.Input(
                                                    id="input-reviews-month",
                                                    type="number",
                                                    min=0,
                                                    step=0.01,
                                                    value=1.2,
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("# propiedades del anfitrión"),
                                                dcc.Input(
                                                    id="input-host-listings",
                                                    type="number",
                                                    min=1,
                                                    value=1,
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            md=3,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(""),
                                                html.Button(
                                                    "PREDECIR",
                                                    id="button-predict",
                                                    n_clicks=0,
                                                    style={
                                                        "marginTop": "18px",
                                                        "width": "100%",
                                                        "backgroundColor": "#0066FF",
                                                        "color": "white",
                                                        "border": "none",
                                                        "borderRadius": "6px",
                                                        "padding": "8px 0",
                                                        "fontWeight": "600",
                                                    },
                                                ),
                                            ],
                                            md=2,
                                        ),
                                    ],
                                    className="gy-2",
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Noches mínimas (también filtra el gráfico inferior)"
                                        ),
                                        dcc.Slider(
                                            id="input-minimum-nights",
                                            min=1,
                                            max=30,
                                            step=1,
                                            value=3,
                                            marks={
                                                1: "1",
                                                7: "7",
                                                14: "14",
                                                30: "30",
                                            },
                                        ),
                                    ],
                                    style={"marginTop": "12px"},
                                ),
                            ],
                        ),
                        # Panel resultados (precio + clasificación)
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        style={
                                            "backgroundColor": "white",
                                            "padding": "12px 16px",
                                            "borderRadius": "8px",
                                            "height": "150px",
                                        },
                                        children=[
                                            html.H6("Precio estimado JPY"),
                                            html.H4(
                                                id="output-price",
                                                children="—",
                                                style={
                                                    "marginTop": "24px",
                                                    "fontWeight": "700",
                                                },
                                            ),
                                        ],
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    html.Div(
                                        style={
                                            "backgroundColor": "white",
                                            "padding": "12px 16px",
                                            "borderRadius": "8px",
                                            "height": "150px",
                                        },
                                        children=[
                                            html.H6("Clasificación: costo por alojamiento"),
                                            html.Div(
                                                id="output-class",
                                                children="Ingresa los datos y presiona PREDECIR.",
                                                style={"marginTop": "16px"},
                                            ),
                                        ],
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-2",
                            style={"marginBottom": "12px"},
                        ),
                        # Scatter
                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "12px 16px",
                                "borderRadius": "8px",
                            },
                            children=[
                                dcc.Graph(
                                    id="scatter-price-capacity",
                                    figure=make_scatter(3),
                                )
                            ],
                        ),
                    ],
                ),
                # --------------------------------------
                # COLUMNA DERECHA (KPIs + EDA)
                # --------------------------------------
                dbc.Col(
                    width=3,
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": "#CCF2FF",
                                "padding": "12px 16px",
                                "borderRadius": "8px",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.H5("tokyo"),
                                html.P(
                                    f"# alojamientos: {num_listings:,.0f}",
                                    style={"marginBottom": "4px"},
                                ),
                                html.P(
                                    f"Precio promedio: {avg_price:,.0f} JPY",
                                    style={"marginBottom": "8px"},
                                ),
                                html.P("Tipos de habitación populares:"),
                                html.Ul(
                                    [
                                        html.Li(f"{idx} ({count} alojamientos)")
                                        for idx, count in top_rooms.items()
                                    ],
                                    style={"marginBottom": "0"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "12px 16px",
                                "borderRadius": "8px",
                                "marginBottom": "12px",
                            },
                            children=[dcc.Graph(id="hist-price", figure=fig_hist)],
                        ),
                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "12px 16px",
                                "borderRadius": "8px",
                            },
                            children=[
                                dcc.Graph(
                                    id="heatmap-precio-reviews",
                                    figure=fig_heatmap,
                                )
                            ],
                        ),
                    ],
                ),
            ]
        ),
    ],
)

# ---------------------------
# CALLBACKS
# ---------------------------

# 1) Actualizar scatter cuando cambia el slider de noches mínimas
@app.callback(
    Output("scatter-price-capacity", "figure"),
    Input("input-minimum-nights", "value"),
)
def update_scatter_callback(min_nights):
    if min_nights is None:
        min_nights = 1
    return make_scatter(min_nights)


# 2) Predicción de precio y clasificación cuando se pulsa PREDECIR
@app.callback(
    [
        Output("output-price", "children"),
        Output("output-class", "children"),
    ],
    Input("button-predict", "n_clicks"),
    State("input-room-type", "value"),
    State("input-accommodates", "value"),
    State("input-availability", "value"),
    State("input-num-reviews", "value"),
    State("input-reviews-month", "value"),
    State("input-host-listings", "value"),
    State("input-minimum-nights", "value"),
)
def hacer_prediccion(
    n_clicks,
    room_type,
    accommodates,
    availability_365,
    num_reviews,
    reviews_per_month,
    host_listings,
    minimum_nights,
):
    if not n_clicks:
        # Antes de oprimir PREDECIR, no cambies nada
        raise PreventUpdate

    valores = [
        room_type,
        accommodates,
        availability_365,
        num_reviews,
        reviews_per_month,
        host_listings,
        minimum_nights,
    ]
    if any(v is None for v in valores):
        return (
            "—",
            "Por favor completa todos los campos antes de predecir.",
        )

    data_input = pd.DataFrame(
        [
            {
                "room_type": room_type,
                "accommodates": float(accommodates),
                "availability_365": float(availability_365),
                "number_of_reviews": float(num_reviews),
                "reviews_per_month": float(reviews_per_month),
                "calculated_host_listings_count": float(host_listings),
                "minimum_nights": float(minimum_nights),
            }
        ]
    )

    # Predicción regresión
    price_pred = reg_pipeline.predict(data_input)[0]
    price_text = f"{price_pred:,.0f} JPY"

    # Predicción clasificación
    class_pred = clf_pipeline.predict(data_input)[0]
    try:
        proba = clf_pipeline.predict_proba(data_input)[0, 1]
    except Exception:
        proba = None

    if class_pred == 1:
        label = "Recomendado (precio competitivo)"
    else:
        label = "No recomendado (precio alto para sus características)"

    if proba is not None:
        class_text = f"{label} – prob. recomendación: {proba:.0%}"
    else:
        class_text = label

    return price_text, class_text


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    # En Dash 3.x se usa app.run en lugar de run_server
    app.run(debug=False, host="0.0.0.0", port=8050)



