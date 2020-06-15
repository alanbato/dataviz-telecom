# Dataset Telefónica
# María José Vota, Eugenia Rendón, Alan Velasco, Martha Elena García
import json

import pandas as pd
import numpy as np
import humanize
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime as dt

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


print("Cargando dataset")
# Initialize data frame
df = pd.read_csv("DataSet_Telefonica.csv")
voronoi_df = pd.read_csv("tower_data.csv")
# eliminar instancias con nulls (20)
df = df.dropna()

print("Limpiando valores")
# convertir formato de fecha: string -> datetime
df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d")
# convertir formato de hora: float -> int
df["hora"] = df["hora"].astype(int)
# Convertir tipos de plan
df["plan"] = df["plan"].astype("category")
print("Construyendo Sankey")
unpopular_plans = list(
    ("GU", "MY", "DD", "KO", "HA", "HW", "NB", "Z2", "DC", "Z7", "DH", "JQ")
)
unpopular_plans.extend(
    ("HX", "HU", "J3", "KB", "FQ", "FD", "HT", "CS", "LQ", "FO", "CH", "L0")
)
unpopular_plans.extend(
    ("FN", "HV", "KK", "BF", "FM", "GG", "FA", "ZR", "HM", "GO", "R2", "HO")
)
unpopular_plans.extend(("ZM", "Q9", "FP", "HH", "HC", "OZ", "KI", "MZ", "GH", "QG"))

df["plan"].cat.remove_categories(unpopular_plans, inplace=True)
df["plan"].cat.add_categories(["Otro"], inplace=True)
df["plan"].fillna("Otro", inplace=True)
# informacion sitio
sitios = df[["sitio", "latitud", "longitud"]].drop_duplicates()


# Data Frame para Sankey

# Agrupar por tecnologia,tipo plan y plan
dfsankey2 = df.groupby(["tecnologia", "tipo_plan"], as_index=False)["sum_bytes"].sum()
dfsankey3 = df.groupby(["tipo_plan", "plan"], as_index=False)["sum_bytes"].sum()
dfsankey2.columns = ["a", "b", "Quantity"]
dfsankey3.columns = ["a", "b", "Quantity"]
dfsankey = dfsankey2.append(dfsankey3)

# Agregar source y target del sankey
all_nodes = dfsankey.a.values.tolist() + dfsankey.b.values.tolist()
source_indices = [all_nodes.index(a) for a in dfsankey.a]
target_indices = [all_nodes.index(b) for b in dfsankey.b]

# graficar sankey
fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=1.0),
                label=all_nodes,
            ),
            link=dict(
                source=source_indices, target=target_indices, value=dfsankey.Quantity,
            ),
        )
    ]
)

fig.update_layout(
    title_text="Por tipo de tecnologia y planes en Telefónica",
    font=dict(size=10, color="white"),
    plot_bgcolor="red",
    paper_bgcolor="#1e1e1e",
)

# first tab histogram analysis
df_Histogram = df.groupby(["tipo_plan"]).agg({"sum_bytes": "sum"})
df_Histogram = df_Histogram.reset_index()

fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        y=df_Histogram["tipo_plan"],
        x=df_Histogram["sum_bytes"],
        orientation="h",
        marker=dict(color="#00ACA8", line=dict(color="#00ACA8", width=1)),
    )
)

fig2.update_layout(
    # title_text="Planes en Telefonica",
    # yaxis_title="Planes",
    font=dict(size=12, color="white"),
    plot_bgcolor="#1e1e1e",
    paper_bgcolor="#1e1e1e",
)

print("Agrupando por semana")
# DataFrame agrupado por semana
grouped = df.groupby(["sitio", "tipo_plan", "tecnologia", "fecha"]).agg(
    {"sum_bytes": "sum"}
)
grouped = grouped.reset_index()
# grouped = grouped.set_index("fecha")
df_master = grouped.groupby(
    ["sitio", "tipo_plan", "tecnologia", pd.Grouper(key="fecha", freq="W-mon")]
).sum()
df_master = df_master.reset_index()
df_master = pd.merge(df_master, sitios, on="sitio", how="left")
df_master = df_master.reset_index()
df = df_master

# Diccionario de ubicaciones importantes en MTY
list_of_locations = {
    "Aeropuerto Internacional MTY": {"lat": 25.7728, "lon": -100.1079},
    "Parque Fundidora": {"lat": 25.6785, "lon": -100.2842},
    "Macroplaza": {"lat": 25.6692, "lon": -100.3099},
    "Paseo Santa Lucia": {"lat": 25.6707, "lon": -100.3059},
    "Cerro de la Silla": {"lat": 25.6320, "lon": -100.2332},
    "Parque la Huasteca": {"lat": 25.6494, "lon": -100.4510},
    "Parque Chipinque": {"lat": 25.6187, "lon": -100.3602},
    "Hospital Universitario": {"lat": 25.6887, "lon": -100.3501},
    "UDEM": {"lat": 25.6609, "lon": -100.4202},
    "Tec de MTY": {"lat": 25.6514, "lon": -100.2895},
}

mapbox_access_token = (
    "pk.eyJ1IjoibWFyaWFqb3NldnoiLCJhIjoiY2s5OTU1OXRq"
    "MDh6bDNubngxaWVyMmZ0aiJ9.2-Wv-0scEzITavhaqSrUcA"
)
app = dash.Dash(__name__)
app.title = "Dash Telefonica"


# Tab Style

tabs_styles = {"height": "44px"}

tab_style = {
    "padding": "6px",
    "fontWeight": "bold",
    "border": "2px solid #302f2f",
    "backgroundColor": "#1e1e1e",
    "color": "white",
}

tab_selected_style = {
    "padding": "6px",
    "border": "2px solid #302f2f",
    "backgroundColor": "#333232",
    "color": "white",
}

company_logo = html.Img(className="logo", src=app.get_asset_url("telefonica-logo.png"))
title = html.H2("Análisis de Tráfico de Datos")
instructions = html.P(
    "Seleccione el lunes de la semana que desea visualizar utilizando el calendario."
)

####Formato Primer Tab
# Colors
bgcolor = "#333232"  # mapbox light map land color

# Figure template
row_heights = [150, 490, 300]
template = {"layout": {"paper_bgcolor": "#1e1e1e", "plot_bgcolor": bgcolor}}


def blank_fig(height, value):
    """
    Build blank figure with the requested height
    """
    return {
        "data": [
            {
                "type": "indicator",
                "value": value,
                "number": {"font": {"color": "white", "size": 40,}},
            }
        ],
        "layout": {
            "template": template,
            "height": 100,
            "margin": {"l": 10, "r": 10, "t": 10, "b": 10},
        },
    }


def build_modal_info_overlay(id, side, content):

    # Build div representing the info overlay for a plot panel
    div = html.Div(
        [  # modal div
            html.Div(
                [  # content div
                    html.Div(
                        [
                            html.H4(
                                [
                                    "Info",
                                    html.Img(
                                        id=f"close-{id}-modal",
                                        src="assets/times-circle-solid.svg",
                                        n_clicks=0,
                                        className="info-icon",
                                        style={"margin": 100},
                                    ),
                                ],
                                className="container_title",
                                style={"color": "white"},
                            ),
                            dcc.Markdown(content),
                        ]
                    )
                ],
                className=f"modal-content {side}",
            ),
            html.Div(className="modal"),
        ],
        id=f"{id}-modal",
        style={"display": "none"},
    )

    return div


# Tab 1 - Analisis
analysis_tab = dcc.Tab(
    label="Análisis ",
    style=tab_style,
    selected_style=tab_selected_style,
    children=[
        html.Div(
            className="row",
            children=[
                html.Div(
                    [
                        html.H1(
                            children=[
                                "Análisis descriptivo",
                                html.A(
                                    html.Img(
                                        className="logo",
                                        src=app.get_asset_url("telefonica-logo.png"),
                                        style={"float": "right", "height": "50px"},
                                    ),
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        # numero de registros
                                        html.H4(
                                            ["Cantidad de registros",],
                                            className="container_title",
                                            style={
                                                "text-align": "center",
                                                "font": {"color": "white", "size": 40,},
                                            },
                                        ),
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="indicator-graph2",
                                                figure=blank_fig(
                                                    row_heights[1], 6831617
                                                ),
                                                config={"displayModeBar": False},
                                            ),
                                            className="svg-container",
                                            style={"height": 250},
                                        ),
                                        # numero de antenas
                                        html.H4(
                                            ["Cantidad de radiobases",],
                                            className="container_title",
                                            style={
                                                "text-align": "center",
                                                "font": {"color": "white", "size": 40,},
                                            },
                                        ),
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="indicator-graph3",
                                                figure=blank_fig(
                                                    row_heights[1],
                                                    df["sitio"].nunique(),
                                                ),
                                                config={"displayModeBar": False},
                                            ),
                                            className="svg-container",
                                            style={"height": 250},
                                        ),
                                        # fecha
                                        html.H4(
                                            ["Fecha de registros"],
                                            className="container_title",
                                            style={
                                                "height": 30,
                                                "text-align": "center",
                                                "font": {"color": "white", "size": 40,},
                                            },
                                        ),
                                        dcc.Markdown(
                                            """ """, style={"height": 10, "padding": 0}
                                        ),
                                        html.P(
                                            "Oct - Nov, 2019",
                                            className="svg-container",
                                            style={
                                                "height": 60,
                                                "padding": 0,
                                                "text-align": "center",
                                                "vertical-align": "baseline",
                                                "justify-content": "center",
                                                "font-size": 40,
                                                "color": "white",
                                            },
                                        ),
                                    ],
                                    className="six columns pretty_container",
                                    id="indicator-div",
                                ),
                                html.Div(
                                    children=[
                                        html.H4(
                                            ["Consumo de datos por tipo de plan",],
                                            className="container_title",
                                        ),
                                        dcc.Graph(
                                            id="radio-histogram",
                                            figure=fig2,
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    className="six columns pretty_container",
                                    id="radio-div",
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.H4(
                                    [
                                        "Distribución de consumo de datos",
                                        html.Img(
                                            id="show-map-modal",
                                            # src="assets/question-circle-solid.svg",
                                            className="info-icon",
                                        ),
                                    ],
                                    className="container_title",
                                ),
                                dcc.Graph(
                                    id="map",
                                    figure=fig,
                                    config={"displayModeBar": False},
                                ),
                            ],
                            className="twelve columns pretty-container",
                            style={"width": "98%", "Margin-left": "10px",},
                            id="map-div",
                        ),
                    ]
                ),
            ],
        ),
    ],
)

# Dropdown para fecha
date_dropdown = html.Div(
    className="div-for-dropdown",
    children=[
        dcc.DatePickerSingle(
            id="date-picker",
            min_date_allowed=dt(2019, 10, 4),
            max_date_allowed=dt(2019, 11, 5),
            initial_visible_month=dt(2019, 10, 4),
            display_format="MMMM D, YYYY",
            style={"border": "0px solid black"},
        )
    ],
)

# Dropdown para lugares conocidos
locations_dropdown = html.Div(
    className="div-for-dropdown",
    children=[
        dcc.Dropdown(
            id="location-dropdown",
            options=[{"label": i, "value": i} for i in list_of_locations],
            placeholder="Seleccione una ubicación",
        )
    ],
)

# Dropdown para tipo de tecnologia
tech_dropdown = html.Div(
    className="div-for-dropdown",
    children=[
        dcc.Dropdown(
            id="tech_name",
            options=[
                {"label": str(b), "value": b} for b in sorted(df["tecnologia"].unique())
            ],
            placeholder="Tipo de tecnología(s)",
        )
    ],
)

# Dropdown para tipo de plan
plan_dropdown = html.Div(
    className="div-for-dropdown",
    children=[
        dcc.Dropdown(
            id="tipo_plan",
            options=[
                {"label": str(b), "value": b} for b in sorted(df["tipo_plan"].unique())
            ],
            placeholder="Tipo de plan(es)",
        )
    ],
)

# Menu con slicers
sidebar_map = html.Div(
    className="four columns div-user-controls",
    children=[
        company_logo,
        title,
        instructions,
        # dropdowns
        html.Div(
            className="row",
            children=[date_dropdown, locations_dropdown, tech_dropdown, plan_dropdown],
        ),
        # ultimas lineas
        dcc.Markdown(children=["TEC de MTY, 2020"]),
    ],
)

# Mapa dinámico
map_graph = html.Div(
    className="eight columns div-for-charts bg-grey",
    children=[
        dcc.Graph(id="map-graph", style={"backgroundColor": "#343332"}),
        html.Div(
            className="text-padding",
            children=html.P("Consumo de datos en zona metropolitana de MTY"),
        ),
        dcc.Graph(id="histogram"),
    ],
)

# Tab 2 - Mapa Dinamico
map_tab = dcc.Tab(
    label="Tráfico de Datos",
    style=tab_style,
    selected_style=tab_selected_style,
    children=[
        # Dash legend - checklists - map
        html.Div(
            className="row",
            children=[
                # Column for user controls
                sidebar_map,
                # Column for app graphs and plots
                map_graph,
            ],
        )
    ],
)

voronoi_graph = html.Div(
    className="div-for-charts bg-grey",
    children=[
        dcc.Graph(
            id="voronoi-graph", style={"backgroundColor": "#343332", "height": "920px"}
        )
    ],
)

voronoi_tab = dcc.Tab(
    label="Cobertura de Antenas",
    style=tab_style,
    selected_style=tab_selected_style,
    children=[html.Div(children=[voronoi_graph])],
)
# Layout of Dash app
app.layout = html.Div(
    children=[
        dcc.Tabs(children=[analysis_tab, map_tab, voronoi_tab], style=tabs_styles)
    ]
)


def get_selection(pickedDate, pickedTech, pickedPlan):
    # Obtener cantidad de bytes por semana.
    # Pinta de otro color la barra seleccionada
    xVal = []
    yVal_control = []
    yVal_prepago = []
    yVal_postpago = []
    colorVal = [
        "#F4EC15",
        "#28C86D",
        "#2E4EA4",
    ]

    # Create new df for sum of weekly sum bytes
    df_subHist = df
    if pickedDate is not None:
        df_subHist = df_subHist[df_subHist["fecha"] == pickedDate]
    if pickedTech is not None:
        df_subHist = df_subHist[df_subHist["tecnologia"] == pickedTech]
    if pickedPlan is not None:
        df_subHist = df_subHist[df_subHist["tipo_plan"] == pickedPlan]

    df_subHist = df_subHist.groupby(["tipo_plan", "tecnologia", "fecha"]).agg(
        {"sum_bytes": "sum"}
    )
    df_subHist = df_subHist.reset_index()

    weeks = df["fecha"].unique()

    # utilizando 4 semanas
    for week in weeks:
        xVal.append(np.datetime_as_string(week, unit="D"))

        yVal_control.append(
            df_subHist[
                (df_subHist["fecha"] == week) & (df_subHist["tipo_plan"] == "Control")
            ]["sum_bytes"].sum()
        )
        yVal_prepago.append(
            df_subHist[
                (df_subHist["fecha"] == week) & (df_subHist["tipo_plan"] == "Prepago")
            ]["sum_bytes"].sum()
        )
        yVal_postpago.append(
            df_subHist[
                (df_subHist["fecha"] == week) & (df_subHist["tipo_plan"] == "Postpago")
            ]["sum_bytes"].sum()
        )

    return [
        np.array(xVal),
        [np.array(yVal_control), [colorVal[0]] * len(weeks)],
        [np.array(yVal_prepago), [colorVal[1]] * len(weeks)],
        [np.array(yVal_postpago), [colorVal[2]] * len(weeks)],
    ]


# Output de histograma
# Update Histogram Figure based on Month, Day and Times Chosen
@app.callback(
    Output("histogram", "figure"),
    [
        Input("date-picker", "date"),
        Input("tech_name", "value"),
        Input("tipo_plan", "value"),
    ],
)
def update_histogram(pickedWeek, pickedTech, pickedPlan):

    xVal, control, prepago, postpago = get_selection(pickedWeek, pickedTech, pickedPlan)
    yVal = np.array(
        [c + p + pp for c, p, pp in zip(control[0], prepago[0], postpago[0])]
    )

    figure = go.Figure(
        data=[
            go.Bar(
                x=xVal,
                y=prepago[0],
                marker_color="#3E4989",  # "#28C86D",
                hoverinfo="y",
                name="Prepago",
            ),
            go.Bar(
                x=xVal,
                y=control[0],
                marker_color="#22928B",  # "#F4EC15",
                hoverinfo="y",
                name="Control",
            ),
            go.Bar(
                x=xVal,
                y=postpago[0],
                marker_color="#B4DE2B",  # "#2E4EA4",
                hoverinfo="y",
                name="Postpago",
            ),
        ],
    )
    figure.update_layout(
        barmode="stack",
        plot_bgcolor="#323130",
        margin=go.layout.Margin(l=50, r=15, t=0, b=0),
        font=dict(color="white"),
        paper_bgcolor="#323130",
        xaxis=dict(type="date", tickangle=0.1),
        yaxis=dict(showgrid=False),
        legend_orientation="h",
        legend=dict(x=-0.056, y=1.2),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=humanize.naturalsize(yi),
                arrowsize=0.3,
                xshift=7,
                standoff=0,
                arrowcolor="#323130",
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    return figure


# Output del mapa
@app.callback(
    Output("map-graph", "figure"),
    [
        Input("date-picker", "date"),
        Input("location-dropdown", "value"),
        Input("tech_name", "value"),
        Input("tipo_plan", "value"),
    ],
)
def update_graph(datePicked, selectedLocation, chosen_tech, chosen_plan):
    df_sub = df
    if datePicked is not None:
        df_sub = df_sub[df_sub["fecha"] == datePicked]
    if chosen_tech is not None:
        df_sub = df_sub[df_sub["tecnologia"] == chosen_tech]
    if chosen_plan is not None:
        df_sub = df_sub[df_sub["tipo_plan"] == chosen_plan]

    df_sub = (
        df_sub.groupby(["sitio", "longitud", "latitud"])
        .agg({"sum_bytes": "sum"})
        .reset_index()
    )

    latInitial = 25.6823
    lonInitial = -100.3030
    zoom = 10.0

    if selectedLocation:
        zoom = 15.0
        latInitial = list_of_locations[selectedLocation]["lat"]
        lonInitial = list_of_locations[selectedLocation]["lon"]

    # Create figure
    return go.Figure(
        data=[
            go.Scattermapbox(
                lon=df_sub["longitud"],
                lat=df_sub["latitud"],
                marker=dict(
                    showscale=True,
                    color=df_sub["sum_bytes"],
                    opacity=1,
                    size=5,
                    colorscale=px.colors.sequential.Viridis,
                    colorbar=dict(
                        title="Consumo<br>Datos",
                        x=0.93,
                        xpad=0,
                        nticks=24,
                        tickfont=dict(color="#d8d8d8"),
                        titlefont=dict(color="#d8d8d8"),
                        thicknessmode="pixels",
                    ),
                ),
                mode="markers",
                hoverinfo="text",
                text=[
                    "Sitio: {} <br> Coord: ({}, {}) <br> Consumo Datos: {}".format(
                        i, j, k, humanize.naturalsize(l)
                    )
                    for i, j, k, l in zip(
                        df["sitio"], df["latitud"], df["longitud"], df_sub["sum_bytes"]
                    )
                ],
            ),
            # Plot important locations on the map
            go.Scattermapbox(
                lat=[list_of_locations[i]["lat"] for i in list_of_locations],
                lon=[list_of_locations[i]["lon"] for i in list_of_locations],
                mode="markers",
                hoverinfo="text",
                text=[i for i in list_of_locations],
                marker=dict(size=8, color="#ffa0a0"),
            ),
        ],
        layout=go.Layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            uirevision="foo",
            clickmode="event+select",
            hovermode="closest",
            hoverdistance=2,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0,
                style="dark",
                center={"lat": latInitial, "lon": lonInitial},
                pitch=40,
                zoom=zoom,
            ),
        ),
    )


# Output del Voronoi


@app.callback(
    Output("voronoi-graph", "figure"),
    [
        Input("date-picker", "date"),
        Input("location-dropdown", "value"),
        Input("tech_name", "value"),
        Input("tipo_plan", "value"),
    ],
)
def update_voronoi(datePicked, selectedLocation, chosen_tech, chosen_plan):
    # Voronoi
    with open("tower_vor.json") as jsonfile:
        geojson = json.load(jsonfile)
    center = {"lat": 25.6823, "lon": -100.3030}
    voronoi_fig = px.choropleth_mapbox(
        voronoi_df,
        geojson=geojson,
        locations="sitio",
        color="normalized",
        hover_data=["consumed_data"],
        featureidkey="properties.tower",
        color_continuous_scale=px.colors.sequential.Viridis,
        range_color=(0, 1),
        center=center,
        mapbox_style="carto-darkmatter",
        zoom=12,
        opacity=0.2,
    )
    voronoi_fig.update_layout(
        autosize=True,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        uirevision="foo",
        clickmode="event+select",
        hovermode="closest",
        # colorbar={"tickfont": {"color": "white"}},
        hoverdistance=2,
        mapbox=dict(accesstoken=mapbox_access_token, bearing=0, pitch=40,),
    )
    return voronoi_fig


if __name__ == "__main__":
    app.run_server(debug=False)
