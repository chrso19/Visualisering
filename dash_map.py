import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import h3
import plotly.express as px
import numpy as np
from geojson import Feature, FeatureCollection
import geojson as json

# -------------------------
# Configuration
# -------------------------
H3_RESOLUTION = 6  # Adjust resolution for hex size (lower = bigger hexes)
DK_CENTER = {"lat": 56.2639, "lon": 11.5018}  # Approx center of Denmark

# -------------------------
# Load data
# -------------------------
data = pd.read_pickle("GBIF_data.pkl")
observations_raw = data["observations"]
plants_raw = data["plants"]
del data

# Filter and select relevant columns
observations = observations_raw[["taxonKey", "decimalLatitude", "decimalLongitude", "year", "month"]]
#observations = observations[(observations["year"] == 2025)]

plants = plants_raw.dropna(subset=["taxonKey", "Danish name", "English name", "Latin name", "Category"])

# -------------------------
# Convert coordinates to H3 hexagons
# -------------------------
def geo_to_h3(row):
    return h3.latlng_to_cell(row.decimalLatitude, row.decimalLongitude, H3_RESOLUTION)

observations['h3_cell'] = observations.apply(geo_to_h3, axis=1)
observations_new = observations[["taxonKey", "h3_cell"]]

# -------------------------
# Aggregate by hexagon
# -------------------------
taxonkey_g = (observations_new
              .groupby('h3_cell')
              .taxonKey
              .agg(list)
              .to_frame("ids")
              .reset_index())

taxonkey_g['count'] = taxonkey_g['ids'].apply(lambda taxonKey: len(taxonKey))
taxonkey_g = taxonkey_g.sort_values('count', ascending=False)

# -------------------------
# Create hexagon geometries (FIXED: swap lat/lng to lng/lat for GeoJSON)
# -------------------------
def add_geometry(row):
    points = h3.cell_to_boundary(row['h3_cell'])
    # GeoJSON requires (longitude, latitude) order, H3 returns (lat, lng)
    return Polygon([(lng, lat) for lat, lng in points])

taxonkey_g['geometry'] = taxonkey_g.apply(add_geometry, axis=1)

# -------------------------
# Convert to GeoJSON (FIXED: use __geo_interface__)
# -------------------------
def hexagons_dataframe_to_geojson(df_hex, hex_id_field, geometry_field, value_field, file_output=None):
    list_features = []
    
    for i, row in df_hex.iterrows():
        feature = Feature(
            geometry=row[geometry_field].__geo_interface__,  # Convert Shapely to dict
            id=row[hex_id_field],
            properties={"value": row[value_field]}
        )
        list_features.append(feature)
    
    feat_collection = FeatureCollection(list_features)
    
    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)
    else:
        return feat_collection

geojson_obj = hexagons_dataframe_to_geojson(
    taxonkey_g,
    hex_id_field='h3_cell',
    value_field='count',
    geometry_field='geometry'
)

# -------------------------
# Create choropleth map
# -------------------------
fig = px.choropleth_map(
    taxonkey_g, 
    geojson=geojson_obj, 
    locations='h3_cell', 
    color='count',
    color_continuous_scale="sunsetdark",
    range_color=(0, taxonkey_g['count'].quantile(0.9)),                  
    map_style='carto-positron',
    zoom=5.5,
    center=DK_CENTER,
    opacity=0.75,
    labels={'count': '# of plant observations'}
)

fig.update_layout(margin={"r": 15, "t": 40, "l": 15, "b": 15})

# -------------------------
# Create Dash app
# -------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Denmark Plant Observations Map", className="text-center mb-4"),
            html.P(f"Showing {len(observations):,} observations across {len(taxonkey_g):,} hexagons (2025 data)", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='plant-map',
                figure=fig,
                style={'height': '80vh'}
            )
        ])
    ])
], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)