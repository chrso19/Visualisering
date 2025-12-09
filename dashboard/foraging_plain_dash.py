import dash
from dash import dcc, html, Input, Output
import pandas as pd
from shapely.geometry import Polygon
import h3
import plotly.express as px
from geojson import Feature, FeatureCollection
import geojson as json

# -------------------------
# colors
# -------------------------
theme = {
    "color1": "#053913",   # black-forest
    "color2": "#942911",   # brandy
    "color3": "#628b48",   # fern
    "color5": "#d58936",   # bronze
        
    "background": "#f7f3e9",   # cream-light
    "text": "#0b0b09",         # onyx
    "header" : "#04250C"   # Dark green
}

# -------------------------
# Configuration
# -------------------------
H3_RESOLUTION = 6
DK_CENTER = {"lat": 56.2639, "lon": 11.5018}
CATEGORIES = ['Herb','Fungi','Flower','Berry','Root','Algae', 'Nut', 'Fruit']

# -------------------------
# Load data
# -------------------------
data = pd.read_pickle("GBIF_data.pkl")
observations_raw = data["observations"]
plants_raw = data["plants"]
del data

# Filter and select relevant columns
observations = observations_raw[["ourID", "decimalLatitude", "decimalLongitude", "year", "month"]]
plants = plants_raw.dropna(subset=["ourID", "Danish name", "English name", "Latin name", "Category"])

# -------------------------
# Convert coordinates to H3 hexagons
# -------------------------
def geo_to_h3(row):
    return h3.latlng_to_cell(row.decimalLatitude, row.decimalLongitude, H3_RESOLUTION)

observations['h3_cell'] = observations.apply(geo_to_h3, axis=1)
observations_new = observations[["ourID", "h3_cell"]]

# -------------------------
# Aggregate by hexagon
# -------------------------
obs_by_hex = (observations_new
              .groupby('h3_cell')
              .ourID
              .agg(list)
              .to_frame("ids")
              .reset_index())

obs_by_hex['count'] = obs_by_hex['ids'].apply(len)
obs_by_hex = obs_by_hex.sort_values('count', ascending=False)

# -------------------------
# Create hexagon geometries
# -------------------------
def add_geometry(row):
    points = h3.cell_to_boundary(row['h3_cell'])
    return Polygon([(lng, lat) for lat, lng in points])

obs_by_hex['geometry'] = obs_by_hex.apply(add_geometry, axis=1)

# -------------------------
# Convert to GeoJSON
# -------------------------
def hexagons_dataframe_to_geojson(df_hex, hex_id_field, geometry_field, value_field):
    list_features = []
    
    for i, row in df_hex.iterrows():
        feature = Feature(
            geometry=row[geometry_field].__geo_interface__,
            id=row[hex_id_field],
            properties={"value": row[value_field]}
        )
        list_features.append(feature)
    
    return FeatureCollection(list_features)

geojson_obj = hexagons_dataframe_to_geojson(
    obs_by_hex,
    hex_id_field='h3_cell',
    value_field='count',
    geometry_field='geometry'
)

# -------------------------
# Create choropleth map
# -------------------------
fig = px.choropleth_map(
    obs_by_hex, 
    geojson=geojson_obj, 
    locations='h3_cell', 
    color='count',
    color_continuous_scale="speed",
    range_color=(0, obs_by_hex['count'].quantile(0.9)),                  
    map_style='carto-positron',
    zoom=5.5,
    center=DK_CENTER,
    opacity=0.75,
    labels={'count': '# of plant observations'}
)

fig.update_layout(
    margin={"r": 15, "t": 40, "l": 15, "b": 15},
    coloraxis_colorbar=dict(
        orientation='h',
        y=-0.15,
        yanchor='top',
        x=0.5,
        xanchor='center',
        len=0.7,
        thickness=15
    )
)

# -------------------------
# Create Dash app
# -------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Denmark Plant Observations", 
                style={'textAlign': 'center', 'marginBottom': '10px'}),
        html.P(f"Showing {len(observations):,} observations across {len(obs_by_hex):,} hexagons", 
               style={'textAlign': 'center', 'color': '#6c757d', 'marginBottom': '30px'})
    ]),
    
    # Main content
    html.Div([
        # Category filters (left sidebar)
        html.Div([
            dcc.Checklist(
                id='category-filter',
                options=[
                    {
                        "label": html.Img(
                            src=f"/assets/icons/{cat}.svg",
                            height=40,
                            style={'display': 'block'},
                            title=cat  # Adds hover tooltip
                        ),
                        "value": cat
                    }
                    for cat in CATEGORIES
                ],
                value=CATEGORIES,  # All selected by default
                labelStyle={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginBottom': '15px',
                    'cursor': 'pointer'
                }
            )
        ], style={
            'width': '80px',
            'padding': '20px 10px',
            'backgroundColor': '#e8f5e9',  # Light green
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        
        # Map
        html.Div([
            dcc.Graph(
                id='plant-map',
                figure=fig,
                style={'height': '75vh'}
            )
        ], style={'width': '55%'}),
        
        # Right column - Info panel
        html.Div([
            html.Div(id='info-panel', children=[
                html.Div([
                    html.H4("Hexagon Details", style={'marginBottom': '20px'}),
                    html.P("Click on a hexagon to see plant observation details", 
                           style={'color': '#6c757d', 'textAlign': 'center', 'marginTop': '100px'})
                ])
            ], style={'position': 'sticky', 'top': '20px'})
        ], style={
            'width': '35%', 
            'borderLeft': '2px solid #dee2e6', 
            'paddingLeft': '20px'
        })
    ], style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start'})
], style={
    'maxWidth': '1800px', 
    'margin': '0 auto', 
    'padding': '20px',
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
})

# -------------------------
# Callback to update map based on category filter
# -------------------------
@app.callback(
    Output('plant-map', 'figure'),
    Input('category-filter', 'value')
)
def update_map(selected_categories):
    if not selected_categories:
        selected_categories = []
    
    # Filter observations based on selected categories
    filtered_plants = plants[plants['Category'].isin(selected_categories)]
    filtered_observations = observations[observations['ourID'].isin(filtered_plants['ourID'])]
    
    if len(filtered_observations) == 0:
        # Return empty map if no data
        empty_fig = px.choropleth_map(
            pd.DataFrame(),
            geojson={"type": "FeatureCollection", "features": []},
            map_style='carto-positron',
            zoom=5.5,
            center=DK_CENTER
        )
        empty_fig.update_layout(margin={"r": 15, "t": 40, "l": 15, "b": 15})
        return empty_fig
    
    # Recalculate hexagon aggregations
    filtered_obs_new = filtered_observations[["ourID", "h3_cell"]]
    filtered_obs_by_hex = (filtered_obs_new
                          .groupby('h3_cell')
                          .ourID
                          .agg(list)
                          .to_frame("ids")
                          .reset_index())
    
    filtered_obs_by_hex['count'] = filtered_obs_by_hex['ids'].apply(len)
    filtered_obs_by_hex = filtered_obs_by_hex.sort_values('count', ascending=False)
    
    # Add geometry
    filtered_obs_by_hex['geometry'] = filtered_obs_by_hex.apply(add_geometry, axis=1)
    
    # Create GeoJSON
    filtered_geojson = hexagons_dataframe_to_geojson(
        filtered_obs_by_hex,
        hex_id_field='h3_cell',
        value_field='count',
        geometry_field='geometry'
    )
    
    # Create new map
    new_fig = px.choropleth_map(
        filtered_obs_by_hex, 
        geojson=filtered_geojson, 
        locations='h3_cell', 
        color='count',
        color_continuous_scale="speed",
        range_color=(0, filtered_obs_by_hex['count'].quantile(0.9)),                  
        map_style='carto-positron',
        zoom=5.5,
        center=DK_CENTER,
        opacity=0.75,
        labels={'count': '# of plant observations'}
    )
    
    new_fig.update_layout(
        margin={"r": 15, "t": 40, "l": 15, "b": 15},
        coloraxis_colorbar=dict(
            orientation='h',
            y=-0.15,
            yanchor='top',
            x=0.5,
            xanchor='center',
            len=0.7,
            thickness=15
        )
    )
    
    return new_fig

# -------------------------
# Callback to update info panel
# -------------------------
@app.callback(
    Output('info-panel', 'children'),
    [Input('plant-map', 'clickData'),
     Input('category-filter', 'value')]
)
def update_info(clickData, selected_categories):
    if clickData is None:
        return html.Div([
            html.H4("Hexagon Details", style={'marginBottom': '20px'}),
            html.P("Click on a hexagon to see plant observation details", 
                   style={'color': '#6c757d', 'textAlign': 'center', 'marginTop': '100px'})
        ])
    
    if not selected_categories:
        selected_categories = []
    
    # Get clicked hexagon ID
    h3_cell = clickData['points'][0]['location']
    
    # Get observations for this hexagon
    hex_observations = observations[observations['h3_cell'] == h3_cell]
    
    # Merge with plants to get species info for each observation
    hex_obs_with_plants = hex_observations.merge(plants, on='ourID', how='left')
    
    # Filter by selected categories
    hex_obs_with_plants = hex_obs_with_plants[hex_obs_with_plants['Category'].isin(selected_categories)]
    
    observation_count = len(hex_obs_with_plants)
    
    if observation_count == 0:
        return html.Div([
            html.H4("Hexagon Details", style={'marginBottom': '20px'}),
            html.Hr(style={'marginBottom': '20px'}),
            html.P("No observations in selected categories for this hexagon", 
                   style={'color': '#6c757d', 'textAlign': 'center', 'marginTop': '50px'})
        ])
    
    # Count unique species
    unique_species = hex_obs_with_plants['English name'].nunique()
    
    # Get top species by observation count
    species_counts = hex_obs_with_plants['English name'].value_counts().head(10)
    
    # Get center coordinates of hexagon
    center = h3.cell_to_latlng(h3_cell)
    
    return html.Div([
        html.H4("Hexagon Details", style={'marginBottom': '20px'}),
        html.Hr(style={'marginBottom': '20px'}),
        
        html.Div([
            html.P([html.Strong("Center: "), f"{center[0]:.4f}°, {center[1]:.4f}°"]),
            html.P([html.Strong("Total Observations: "), f"{observation_count:,}"]),
            html.P([html.Strong("Unique Species: "), f"{unique_species:,}"]),
        ], style={'marginBottom': '30px'}),
        
        html.H5("Top 10 Species", style={'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.Div([
                    html.Strong(species, style={'fontSize': '14px'}),
                    html.Span(f" ({count})", 
                             style={'color': '#6c757d', 'marginLeft': '8px', 'fontSize': '13px'})
                ], style={'marginBottom': '10px'})
                for species, count in species_counts.items()
            ])
        ], style={'maxHeight': '400px', 'overflowY': 'auto'})
    ], style={'padding': '15px'})

if __name__ == '__main__':
    app.run(debug=True)