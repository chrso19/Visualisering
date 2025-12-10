import dash
from dash import dcc, html, Input, Output, State, ALL
import pandas as pd
from shapely.geometry import Polygon
import h3
import plotly.express as px
import plotly.graph_objects as go
from geojson import Feature, FeatureCollection
import geojson as json
from datetime import date


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
observations = observations_raw[["ourID", "decimalLatitude", "decimalLongitude", "eventDate"]].copy()
plants = plants_raw.dropna(subset=["ourID", "Danish name", "English name", "Latin name", "Category"])

# -------------------------
# Set up for datepicker
# -------------------------

observations['eventDate'] = pd.to_datetime(observations['eventDate'], errors='coerce')
max_event = observations['eventDate'].max()
min_event = observations['eventDate'].min()

# Default values for datepicker (1 year from latest event)
default_end = max_event.date().isoformat() if pd.notnull(max_event) else None  
default_start = ((max_event - pd.DateOffset(years=1)).date().isoformat() if pd.notnull(max_event) else None)

# Ensure that user can only select dates within the range of our data
min_date_allowed = min_event.date().isoformat() if pd.notnull(min_event) else None
max_date_allowed = default_end

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
    """Convert H3 cell to Shapely Polygon geometry"""
    points = h3.cell_to_boundary(row['h3_cell'])
    return Polygon([(lng, lat) for lat, lng in points])

obs_by_hex['geometry'] = obs_by_hex.apply(add_geometry, axis=1)

# -------------------------
# Convert to GeoJSON
# -------------------------
def hexagons_dataframe_to_geojson(df_hex, hex_id_field, geometry_field, value_field):
    """Convert hexagon dataframe to GeoJSON FeatureCollection"""
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
# Map configuration constants
# -------------------------
# Default map parameters used when creating/updating map
MAP_CONFIG = {
    'color' : 'count',
    'color_continous_scale' : 'speed',
    'map_style' : 'carto-positron',
    'center' : DK_CENTER,
    'opacity' : 0.75,
    'labels' : {'count': '# of plant observations'},
    'margin' : {"r": 15, "t": 40, "l": 15, "b": 15},
    'colorbar_config' : dict(
        orientation='h',
        y=-0.15,
        yanchor='top',
        x=0.5,
        xanchor='center',
        len=0.7,
        thickness=15
    )
}

# -------------------------
# Create initial choropleth map
# -------------------------
fig = px.choropleth_map(
    obs_by_hex, 
    geojson=geojson_obj, 
    locations='h3_cell', 
    color=MAP_CONFIG['color'],
    color_continuous_scale=MAP_CONFIG['color_continous_scale'],
    range_color=(0, obs_by_hex['count'].quantile(0.9)),                  
    map_style=MAP_CONFIG['map_style'],
    zoom=5.5,
    center=MAP_CONFIG['center'],
    opacity=MAP_CONFIG['opacity'],
    labels=MAP_CONFIG['labels']
)

fig.update_layout(
    margin=MAP_CONFIG['margin'],
    coloraxis_colorbar=MAP_CONFIG['colorbar_config']
)

# -------------------------
# Create Dash app
# -------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Foraging map of Denmark", 
                style={'textAlign': 'center', 'marginBottom': '10px'})
    ]),
    
    # Date range filter for the map (aligned left, above sidebar and map)
    html.Div([
        dcc.DatePickerRange(
            id='date-filter',
            start_date=default_start,
            end_date=default_end,
            min_date_allowed=min_date_allowed,
            max_date_allowed=max_date_allowed,
            display_format='YYYY-MM-DD',
            minimum_nights=0
        )
    ], style={'textAlign': 'left', 'marginBottom': '20px'}),
    
    # Hidden store to track the currently clicked hexagon
    dcc.Store(id='clicked-hexagon', data=None),
    
    # Main content
    html.Div([
        
        # Category buttons (left sidebar)
        html.Div([
            # Store for active categories
            dcc.Store(id='active-categories', data=CATEGORIES),
            html.Div(
                [
                    # Create category filter buttons using list comprehension, * is for unpacking list
                    *[
                        html.Button(
                            [
                                html.Img(src=f"/assets/icons/{cat}.svg", height=40, 
                                        style={'verticalAlign': 'middle', 'marginRight': '8px'}),
                                html.Span(cat, style={'verticalAlign': 'middle'})
                            ],
                            id={"type": "cat-btn", "cat": cat},
                            n_clicks=1,
                            className='cat-button selected',
                            style={
                                'width': '120px',
                                'height': '48px',
                                'display': 'flex',
                                'alignItems': 'left',
                                'justifyContent': 'center',
                                'padding': '0',
                                'boxSizing': 'border-box'
                            }
                        )
                        for cat in CATEGORIES
                    ], 

                    # Select all / Deselect all button
                    html.Div([
                        html.Button(id='select-all-btn', n_clicks=0, className='btn-select')
                    ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '8px', 'marginTop': '12px'})
                ],
                style={'display': 'flex', 'flexDirection': 'column', 'gap': '8px', 'alignItems': 'center'}
            )
    
        ], style={
            'width': '80px',
            'padding': '20px 10px',
            'backgroundColor': '#e8f5e9',
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
# Callback to update select all/deselect all button text
# -------------------------
@app.callback(
    Output('select-all-btn', 'children'),
    [Input('active-categories', 'data')]
)
def update_select_all_text(active_cats):
    """Update button text based on whether all categories are selected"""
    if active_cats is None:
        active_cats = []
    if set(active_cats) == set(CATEGORIES):
        return "Deselect all"
    else:
        return "Select all"

# -------------------------
# Callback to manage category selection state
# -------------------------
@app.callback(
    Output("active-categories", "data"),
    Input({"type": "cat-btn", "cat": ALL}, "n_clicks"),
    Input('select-all-btn', 'n_clicks'),
    State("active-categories", "data"),
    State({"type": "cat-btn", "cat": ALL}, "id"),
)
def toggle_categories(n_clicks_list, select_all_clicks, active_cats, ids):
    """Manage which categories are currently active based on button clicks"""
    
    # Find which button triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return active_cats

    triggered = ctx.triggered_id
    
    # Initialize empty list if needed
    if active_cats is None:
        active_cats = []

    # Handle Select all/Deselect all button
    if isinstance(triggered, str) and triggered == 'select-all-btn':
        if set(active_cats) == set(CATEGORIES):
            return []  # Deselect all
        else:
            return CATEGORIES  # Select all

    # Handle individual category button toggle
    if isinstance(triggered, dict):
        cat = triggered.get('cat')
        
        # Just for robustness
        if cat is None:
            return active_cats

        # Toggle category: remove if active, add if inactive
        if cat in active_cats:
            return [c for c in active_cats if c != cat]
        else:
            return active_cats + [cat]

    # Fallback
    return active_cats

# -------------------------
# Callback to update button visual styles active/inactive
# -------------------------
@app.callback(
    Output({"type": "cat-btn", "cat": ALL}, "className"),
    [Input('active-categories', 'data')]
)
def update_button_classes(selected):
    """Update CSS classes for category buttons based on selection state"""
    if selected is None:
        selected = []
    return ['cat-button selected' if cat in selected else 'cat-button' for cat in CATEGORIES]

# -------------------------
# Callback to store clicked hexagon
# -------------------------
@app.callback(
    Output('clicked-hexagon', 'data'),
    Input('plant-map', 'clickData')
)
def store_clicked_hexagon(clickData):
    """Store the ID of the clicked hexagon for highlighting"""
    if clickData is None:
        return None
    return clickData['points'][0]['location']

# -------------------------
# Callback to update map based on filters and highlight clicked hexagon
# -------------------------
@app.callback(
    Output('plant-map', 'figure'),
    [Input('active-categories', 'data'),
     Input('date-filter', 'start_date'),
     Input('date-filter', 'end_date'),
     Input('clicked-hexagon', 'data')]
)
def update_map(selected_categories, start_date, end_date, clicked_hex):
    """Update map based on category and date filters, and highlight selected hexagon"""
    if not selected_categories:
        selected_categories = []
    
    # Filter observations based on selected categories
    filtered_plants = plants[plants['Category'].isin(selected_categories)]
    filtered_observations = observations[observations['ourID'].isin(filtered_plants['ourID'])]

    # Filter by date range
    if start_date:
        try:
            start_ts = pd.to_datetime(start_date)
            filtered_observations = filtered_observations[filtered_observations['eventDate'] >= start_ts]
        except Exception:
            pass
    if end_date:
        try:
            end_ts = pd.to_datetime(end_date)
            filtered_observations = filtered_observations[filtered_observations['eventDate'] <= end_ts]
        except Exception:
            pass
    
    if len(filtered_observations) == 0:
        # Return empty map if no data
        empty_fig = px.choropleth_map(
            pd.DataFrame(),
            geojson={"type": "FeatureCollection", "features": []},
            map_style='carto-positron',
            zoom=5.5,
            center=DK_CENTER
        )
        # Add invisible marker to maintain colorbar
        empty_fig.add_trace(go.Scattermapbox(
            lat=[DK_CENTER['lat']],
            lon=[DK_CENTER['lon']],
            mode='markers',
            marker=dict(
                size=0,
                color=[0],
                colorscale=MAP_CONFIG['color_continous_scale'],
                cmin=0,
                cmax=1,
                colorbar=MAP_CONFIG['colorbar_config'],
                showscale=True
            ),
            showlegend=False
        ))
        empty_fig.update_layout(margin={"r": 15, "t": 40, "l": 15, "b": 15},
                                mapbox_style=MAP_CONFIG['map_style'],
                                mapbox_center=MAP_CONFIG['center'],
                                mapbox_zoom=5.5)
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
        color=MAP_CONFIG['color'],
        color_continuous_scale=MAP_CONFIG['color_continous_scale'],
        range_color=(0, filtered_obs_by_hex['count'].quantile(0.9)),                  
        map_style=MAP_CONFIG['map_style'],
        zoom=5.5,
        center=MAP_CONFIG['center'],
        opacity=MAP_CONFIG['opacity'],
        labels=MAP_CONFIG['labels']
    )
    
    # Update layout
    new_fig.update_layout(
        margin=MAP_CONFIG['margin'],
        coloraxis_colorbar=MAP_CONFIG['colorbar_config']
    )
    
    # Add marker for clicked hexagon
    if clicked_hex is not None and clicked_hex in filtered_obs_by_hex['h3_cell'].values:
        try:
            # Get center coordinates of clicked hexagon
            center_lat, center_lon = h3.cell_to_latlng(clicked_hex)
            
            # Add a marker at the center
            new_fig.add_scattermap(
                lat=[center_lat],
                lon=[center_lon],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='circle',
                    opacity=0.8
                ),
                showlegend=False,
                hoverinfo='skip'
            )
        except Exception as e:
            print(f"Marker error: {e}")
            pass
    
    return new_fig

# -------------------------
# Callback to update info panel with details and mini-map
# -------------------------
@app.callback(
    Output('info-panel', 'children'),
    [Input('plant-map', 'clickData'),
     Input('active-categories', 'data'),
     Input('date-filter', 'start_date'),
     Input('date-filter', 'end_date')]
)
def update_info_panel(clickData, selected_categories, start_date, end_date):
    """Update info panel with hexagon details, mini-map, and species list"""
    if clickData is None:
        return html.Div([
            html.H4("Hexagon Details", style={'marginBottom': '20px'}),
            html.P("Click on a hexagon to see plant observation details", 
                   style={'color': '#6c757d', 'textAlign': 'center', 'marginTop': '100px'})
        ])
    
    if not selected_categories:
        selected_categories = []

    # Get clicked hexagon and its neighbors
    h3_cell = clickData['points'][0]['location']
    neighbors = set(h3.grid_ring(h3_cell, 1))
    neighbors.add(h3_cell)  # Include the clicked hexagon itself

    # Get observations within hexagon and neighbors
    hex_observations = observations[observations['h3_cell'].isin(neighbors)].copy()

    # Apply date filter
    if start_date:
        try:
            hex_observations = hex_observations[hex_observations['eventDate'] >= pd.to_datetime(start_date)]
        except:
            pass
    if end_date:
        try:
            hex_observations = hex_observations[hex_observations['eventDate'] <= pd.to_datetime(end_date)]
        except:
            pass

    # Merge with plant data to get species information
    hex_obs_with_plants = hex_observations.merge(plants, on='ourID', how='left')
    hex_obs_with_plants = hex_obs_with_plants[hex_obs_with_plants['Category'].isin(selected_categories)]
    
    observation_count = len(hex_obs_with_plants)
    if observation_count == 0:
        return html.Div([
            html.H4("Hexagon Details"),
            html.P("No observations in selected categories for this hexagon", 
                   style={'color': '#6c757d', 'textAlign': 'center', 'marginTop': '50px'})
        ])
    
    # Calculate statistics
    unique_species = hex_obs_with_plants['English name'].nunique()
    center = h3.cell_to_latlng(h3_cell)

    # ----------------------------
    # Create mini-map of observations
    # ----------------------------
    if not hex_obs_with_plants.empty:
        fig = px.scatter_map(
            hex_obs_with_plants,
            lat='decimalLatitude',
            lon='decimalLongitude',
            color='Category',
            hover_name='English name',
            zoom=10,
            height=250,
            center={"lat": center[0], "lon": center[1]}  # Center on clicked hexagon
        )

        # Add semi-transparent polygon for the clicked hexagon
        hex_geom = add_geometry({'h3_cell': h3_cell})
        fig.add_scattermap(
            lat=[pt[1] for pt in hex_geom.exterior.coords],
            lon=[pt[0] for pt in hex_geom.exterior.coords],
            mode='lines',
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0.5)', width=2),
            showlegend=False,
            hoverinfo='skip'
        )

        fig.update_layout(
            map_style='carto-positron', 
            margin=dict(l=0, r=0, t=0, b=0)
        )
    else:
        fig = go.Figure()

    # Get top 10 species by count
    species_counts = hex_obs_with_plants['English name'].value_counts().head(10)
    
    return html.Div([
        # Header with hexagon center coordinates
        html.Div([
            html.H4("Hexagon Details", style={'display': 'inline-block', 'marginRight': '20px'}),
            html.Span(f"Center: {center[0]:.4f}°, {center[1]:.4f}°", 
                     style={'fontSize': '14px', 'color': '#555'})
        ], style={'marginBottom': '10px'}),
        
        # Statistics
        html.Div([
            html.Span(f"Total Observations: {observation_count}", 
                     style={'marginRight': '15px', 'fontSize': '13px'}),
            html.Span(f"Unique Species: {unique_species}", 
                     style={'fontSize': '13px'})
        ], style={'marginBottom': '15px'}),
        
        # Mini-map
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        
        # Top 10 species list
        html.H5("Top 10 Species", style={'marginTop': '15px', 'marginBottom': '10px'}),
        html.Div([
            html.Div([
                html.Strong(species, style={'fontSize': '12px'}),
                html.Span(f" ({count})", 
                         style={'color': '#6c757d', 'marginLeft': '6px', 'fontSize': '11px'})
            ], style={'marginBottom': '6px'})
            for species, count in species_counts.items()
        ], style={'maxHeight': '200px', 'overflowY': 'auto'})
    ], style={'padding': '10px'})


if __name__ == '__main__':
    app.run(debug=True)