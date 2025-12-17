import dash
from dash import dcc, html, Input, Output, State, ALL
import pandas as pd
from shapely.geometry import Polygon
import h3
import plotly.express as px
import plotly.graph_objects as go
from geojson import Feature, FeatureCollection


# -------------------------
# Configuration
# -------------------------
H3_RESOLUTION = 6
DK_CENTER = {"lat": 56.2639, "lon": 11.5018}
CATEGORIES = ['Herb','Fungi','Flower','Berry','Root','Algae', 'Nut', 'Fruit']
CATEGORY_COLORS = {
    "Algae":  "#11bf40",  
    "Flower": "#bab823",  
    "Herb":   "#628b48",  
    "Fruit":   "#9e1210",  
    "Root":  "#e3421e",  
    "Berry":  "#690c4a",  
    "Nut":  "#e6a763",  
    "Fungi":    "#5c1c0d",  
}

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

# -------------------------
# Helper functions
# -------------------------
def add_geometry(row):
    """Convert H3 cell to Shapely Polygon geometry"""
    points = h3.cell_to_boundary(row['h3_cell'])
    return Polygon([(lng, lat) for lat, lng in points])

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

def aggregate_by_hex(filtered_observations):
    """Aggregate observations by hexagon and add geometry"""
    filtered_obs_new = filtered_observations[["ourID", "h3_cell"]]
    filtered_obs_by_hex = (filtered_obs_new
                          .groupby('h3_cell')
                          .ourID
                          .agg(list)
                          .to_frame("ids")
                          .reset_index())
    
    filtered_obs_by_hex['count'] = filtered_obs_by_hex['ids'].apply(len)
    filtered_obs_by_hex = filtered_obs_by_hex.sort_values('count', ascending=False)
    filtered_obs_by_hex['geometry'] = filtered_obs_by_hex.apply(add_geometry, axis=1)
    
    return filtered_obs_by_hex

def apply_date_filter(df, start_date, end_date):
    """Apply date range filter to observations dataframe"""
    filtered = df.copy()
    
    if start_date:
        try:
            start_ts = pd.to_datetime(start_date)
            filtered = filtered[filtered['eventDate'] >= start_ts]
        except Exception:
            pass
    if end_date:
        try:
            end_ts = pd.to_datetime(end_date)
            filtered = filtered[filtered['eventDate'] <= end_ts]
        except Exception:
            pass
    
    return filtered

# -------------------------
# Initial aggregation
# -------------------------
obs_by_hex = aggregate_by_hex(observations)

# -------------------------
# Convert to GeoJSON
# -------------------------
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
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Foraging map of Denmark", 
                style={'textAlign': 'center', 'marginBottom': '10px'})
    ]),
    
    # Date range filter for the map
    html.Div([
        dcc.DatePickerRange(
            id='date-filter',
            start_date=default_start,
            end_date=default_end,
            min_date_allowed=min_date_allowed,
            max_date_allowed=max_date_allowed,
            display_format='YYYY-MM-DD',
            minimum_nights=0,
            style={'fontSize': '12px'}
        )
    ], style={'textAlign': 'left', 'marginBottom': '15px'}),
    
    # Hidden stores
    dcc.Store(id='clicked-hexagon', data=None), # Which hexagon is currently clicked
    dcc.Store(id='active-categories', data=CATEGORIES), #categories that are selected
    dcc.Store(id='filtered-hex-data', data=None),  # Store filtered data to avoid recalculation
    dcc.Store(id='back-button-clicks', data=0), # for the "back to top 10 species"
    
    # Main content
    html.Div([
        # Category buttons (left sidebar)
        html.Div([
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
                                'alignItems': 'center',
                                'justifyContent': 'flex-start',
                                'padding': '0 8px',
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
            'padding': '20px 10px'
        }),
        
        # Map
        html.Div([
            dcc.Graph(
                id='plant-map',
                figure=fig,
                style={'height': '75vh'},
                config={'doubleClick': 'reset'}
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

    return active_cats

# -------------------------
# Callback to update button visual styles (active/inactive)
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
# Callback to store filtered hex data
# -------------------------
@app.callback(
    Output('filtered-hex-data', 'data'),
    [Input('active-categories', 'data'),
     Input('date-filter', 'start_date'),
     Input('date-filter', 'end_date')]
)
def store_filtered_data(selected_categories, start_date, end_date):
    """Store filtered observations to avoid recalculating in multiple callbacks"""
    if not selected_categories:
        return None
    
    # Filter observations based on selected categories
    filtered_plants = plants[plants['Category'].isin(selected_categories)]
    filtered_observations = observations[observations['ourID'].isin(filtered_plants['ourID'])]
    
    # Filter by date range
    filtered_observations = apply_date_filter(filtered_observations, start_date, end_date)
    
    if len(filtered_observations) == 0:
        return None
    
    # Store as dictionary with the data we need
    return {
        'h3_cells': filtered_observations['h3_cell'].tolist(),
        'ourIDs': filtered_observations['ourID'].tolist(),
        'lats': filtered_observations['decimalLatitude'].tolist(),
        'lons': filtered_observations['decimalLongitude'].tolist(),
        'dates': filtered_observations['eventDate'].astype(str).tolist()
    }

# -------------------------
# Callback to update map
# -------------------------
@app.callback(
    Output('plant-map', 'figure'),
    [Input('filtered-hex-data', 'data'),
     Input('clicked-hexagon', 'data')],
    [State('plant-map', 'relayoutData')]
)
def update_map(filtered_data, clicked_hex, relayoutData):
    """Update map based on filtered data and highlight selected hexagon"""
    
    # Extract current zoom and center from relayoutData
    zoom = 5.5
    center = DK_CENTER
    
    if relayoutData:
        if 'mapbox.zoom' in relayoutData:
            zoom = relayoutData['mapbox.zoom']
        if 'mapbox.center' in relayoutData:
            center = relayoutData['mapbox.center']

    if filtered_data is None:
        # Return empty map if no data
        empty_fig = px.choropleth_map(
            pd.DataFrame(),
            geojson={"type": "FeatureCollection", "features": []},
            map_style='carto-positron',
            zoom=zoom,
            center=center
        )
        # Add invisible marker to maintain colorbar
        empty_fig.add_trace(go.Scattermapbox(
            lat=[center['lat']],
            lon=[center['lon']],
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
        empty_fig.update_layout(
            margin=MAP_CONFIG['margin'],
            mapbox_style=MAP_CONFIG['map_style'],
            mapbox_center=center,
            mapbox_zoom=zoom
        )
        return empty_fig
    
    # Reconstruct filtered observations from stored data
    filtered_observations = pd.DataFrame({
        'h3_cell': filtered_data['h3_cells'],
        'ourID': filtered_data['ourIDs']
    })
    
    # Recalculate hexagon aggregations
    filtered_obs_by_hex = aggregate_by_hex(filtered_observations)
    
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
        zoom=zoom,
        center=center,
        opacity=MAP_CONFIG['opacity'],
        labels=MAP_CONFIG['labels']
    )
    
    # Update layout
    new_fig.update_layout(
        margin=MAP_CONFIG['margin'],
        coloraxis_colorbar=MAP_CONFIG['colorbar_config'],
        uirevision='constant'
    )
    
    # Add marker for clicked hexagon
    if clicked_hex is not None and clicked_hex in filtered_obs_by_hex['h3_cell'].values:
        try:
            center_lat, center_lon = h3.cell_to_latlng(clicked_hex)
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
    
    return new_fig

# -------------------------
# Callback to update info panel
# -------------------------
@app.callback(
    Output('info-panel', 'children'),
    [Input('plant-map', 'clickData'),
     Input('active-categories', 'data'),
     Input('filtered-hex-data', 'data')]
)
def update_info_panel(clickData, selected_categories, filtered_data):
    """Update info panel with hexagon details, mini-map, and species list"""
    if clickData is None:
        return html.Div([
            html.H4("Hexagon Details", style={'marginBottom': '20px'}),
            html.P("Click on a hexagon to see plant observation details", 
                   style={'color': '#6c757d', 'textAlign': 'center', 'marginTop': '100px'})
        ])
    
    if not selected_categories or filtered_data is None:
        return html.Div([
            html.H4("Hexagon Details"),
            html.P("No observations in selected categories", 
                   style={'color': '#6c757d', 'textAlign': 'center', 'marginTop': '50px'})
        ])

    h3_cell = clickData['points'][0]['location']
    
    # Reconstruct filtered observations
    filtered_observations = pd.DataFrame({
        'h3_cell': filtered_data['h3_cells'],
        'ourID': filtered_data['ourIDs'],
        'decimalLatitude': filtered_data['lats'],
        'decimalLongitude': filtered_data['lons'],
        'eventDate': pd.to_datetime(filtered_data['dates'])
    })
    
    # Get observations within ONLY the clicked hexagon (for stats)
    hex_observations_stats = filtered_observations[filtered_observations['h3_cell'] == h3_cell].copy()
    
    # Merge with plant data for stats
    hex_obs_with_plants_stats = hex_observations_stats.merge(plants, on='ourID', how='left')
    hex_obs_with_plants_stats = hex_obs_with_plants_stats[hex_obs_with_plants_stats['Category'].isin(selected_categories)]
    
    observation_count = len(hex_obs_with_plants_stats)
    unique_species = hex_obs_with_plants_stats['ourID'].nunique()
    
    if observation_count == 0:
        return html.Div([
            html.H5("Hexagon Details"),
            html.P("No observations in selected categories for this hexagon", 
                   style={'color': '#6c757d', 'textAlign': 'center', 'marginTop': '50px'})
        ])
    
    center = h3.cell_to_latlng(h3_cell)
    
    # Get neighbors for mini-map display
    neighbors = set(h3.grid_ring(h3_cell, 1))
    neighbors.add(h3_cell)
    
    # Get observations within hexagon and neighbors (for mini-map)
    hex_observations = filtered_observations[filtered_observations['h3_cell'].isin(neighbors)].copy()
    hex_obs_with_plants = hex_observations.merge(plants, on='ourID', how='left')
    hex_obs_with_plants = hex_obs_with_plants[hex_obs_with_plants['Category'].isin(selected_categories)]

    # Create mini-map
    if not hex_obs_with_plants.empty:
        fig = px.scatter_map(
            hex_obs_with_plants,
            lat='decimalLatitude',
            lon='decimalLongitude',
            color='Category',
            hover_name='English name',
            zoom=10,
            height=250,
            center={"lat": center[0], "lon": center[1]}
        )

        # Add hexagon outline
        hex_geom = add_geometry({'h3_cell': h3_cell})
        fig.add_scattermap(
            lat=[pt[1] for pt in hex_geom.exterior.coords],
            lon=[pt[0] for pt in hex_geom.exterior.coords],
            mode='lines',
            fill='toself',
            fillcolor='rgba(255,0,0,0.08)',
            line=dict(color='rgba(255,0,0,0.3)', width=1.5),
            showlegend=False,
            hoverinfo='skip'
        )

        fig.update_layout(
            map_style='carto-positron', 
            margin=dict(l=0, r=0, t=0, b=0)
        )
    else:
        fig = go.Figure()

    # Get top 10 species by count (from clicked hexagon only)
    species_counts = hex_obs_with_plants_stats['English name'].value_counts().head(10)
    
    return html.Div([
        # Header with hexagon center coordinates
        html.Div([
            html.H5("Hexagon Details", style={'display': 'inline-block', 'marginRight': '15px', 'marginBottom': '0'}),
            html.Span(f"Center: {center[0]:.4f}°, {center[1]:.4f}°", 
                     style={'fontSize': '12px', 'color': '#555'}),
            html.Span(" | ", style={'margin': '0 8px', 'color': '#dee2e6'}),
            html.Span(f"Observations: {observation_count}", 
                     style={'fontSize': '12px'}),
            html.Span(" | ", style={'margin': '0 8px', 'color': '#dee2e6'}),
            html.Span(f"Species: {unique_species}", 
                     style={'fontSize': '12px'})
        ], style={'marginBottom': '10px'}),
        
        # Mini-map with clickable observations
        dcc.Graph(id='mini-map', figure=fig, config={'displayModeBar': False}),
        
        # Placeholder for observation details
        html.Div(id='observation-details', children=[
            html.H5("Top 10 Species", style={'marginTop': '15px', 'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Strong(species, style={'fontSize': '12px'}),
                    html.Span(f" ({count})", 
                             style={'color': '#6c757d', 'marginLeft': '6px', 'fontSize': '11px'})
                ], style={'marginBottom': '6px'})
                for species, count in species_counts.items()
            ], style={'maxHeight': '200px', 'overflowY': 'auto'})
        ])
    ], style={'padding': '10px'})

# -------------------------
# Callback to handle back button clicks
# -------------------------
@app.callback(
    Output('back-button-clicks', 'data'),
    Input({'type': 'back-btn', 'index': ALL}, 'n_clicks'),
    State('back-button-clicks', 'data'),
    prevent_initial_call=True
)
def handle_back_button(n_clicks_list, current_count):
    """Increment counter when back button is clicked"""
    if any(n_clicks_list):
        return (current_count or 0) + 1
    return current_count or 0

# -------------------------
# Callback to update mini-map
# -------------------------
@app.callback(
    Output('mini-map', 'figure'),
    [Input('plant-map', 'clickData'),
     Input('filtered-hex-data', 'data')]
)
def update_minimap(main_click, filtered_data):
    """Update mini-map with enhanced hover information"""
    if main_click is None or filtered_data is None:
        return go.Figure()
    
    h3_cell = main_click['points'][0]['location']
    neighbors = set(h3.grid_ring(h3_cell, 1))
    neighbors.add(h3_cell)

    # Reconstruct filtered observations
    filtered_observations = pd.DataFrame({
        'h3_cell': filtered_data['h3_cells'],
        'ourID': filtered_data['ourIDs'],
        'decimalLatitude': filtered_data['lats'],
        'decimalLongitude': filtered_data['lons'],
        'eventDate': pd.to_datetime(filtered_data['dates'])
    })
    
    hex_observations = filtered_observations[filtered_observations['h3_cell'].isin(neighbors)].copy()
    hex_obs_with_plants = hex_observations.merge(plants, on='ourID', how='left')
    
    if hex_obs_with_plants.empty:
        return go.Figure()
    
    # Format date for display
    hex_obs_with_plants['formatted_date'] = hex_obs_with_plants['eventDate'].dt.strftime('%Y-%m-%d')
    hex_obs_with_plants['formatted_date'] = hex_obs_with_plants['formatted_date'].fillna('Unknown')
    
    center = h3.cell_to_latlng(h3_cell)
    
    fig = px.scatter_map(
        hex_obs_with_plants,
        lat='decimalLatitude',
        lon='decimalLongitude',
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        category_orders={"Category": CATEGORIES},
        hover_name='English name',
        hover_data={
            'Category': True,
            'formatted_date': True,
            'decimalLatitude': False,
            'decimalLongitude': False
        },
        custom_data=['ourID'],
        zoom=10,
        height=250,
        center={"lat": center[0], "lon": center[1]}
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                      'Category: %{customdata[1]}<br>' +
                      'Date: %{customdata[2]}<br>' +
                      '<extra></extra>',
        customdata=hex_obs_with_plants[['ourID', 'Category', 'formatted_date']].values
    )
    
    fig.update_layout(
        map_style='carto-positron', 
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # Add hexagon outline
    hex_geom = add_geometry({'h3_cell': h3_cell})
    fig.add_scattermap(
        lat=[pt[1] for pt in hex_geom.exterior.coords],
        lon=[pt[0] for pt in hex_geom.exterior.coords],
        mode='lines',
        fill='toself',
        fillcolor='rgba(255,0,0,0.08)',
        line=dict(color='rgba(255,0,0,0.3)', width=1.5),
        showlegend=False,
        hoverinfo='skip'
    )
    
    return fig

# -------------------------
# Callback to update observation details when clicking on mini-map
# -------------------------
@app.callback(
    Output('observation-details', 'children'),
    [Input('mini-map', 'clickData'),
     Input('back-button-clicks', 'data')],
    [State('plant-map', 'clickData'),
     State('filtered-hex-data', 'data')]
)
def update_observation_details(minimap_click, back_count, main_click, filtered_data):
    """Update observation details when clicking on mini-map"""
    
    # Check which input triggered the callback
    ctx = dash.callback_context
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'back-button-clicks':
            minimap_click = None
    
    if main_click is None or filtered_data is None:
        return html.Div()
    
    h3_cell = main_click['points'][0]['location']
    
    # Reconstruct filtered observations
    filtered_observations = pd.DataFrame({
        'h3_cell': filtered_data['h3_cells'],
        'ourID': filtered_data['ourIDs'],
        'decimalLatitude': filtered_data['lats'],
        'decimalLongitude': filtered_data['lons'],
        'eventDate': pd.to_datetime(filtered_data['dates'])
    })
    
    # Include the main hexagon and its neighbors for observation lookup
    neighbors = set(h3.grid_ring(h3_cell, 1))
    neighbors.add(h3_cell)
    
    # Get observations for the clicked hexagon AND its neighbors (to match minimap content)
    hex_observations = filtered_observations[filtered_observations['h3_cell'].isin(neighbors)].copy()
    hex_obs_with_plants = hex_observations.merge(plants, on='ourID', how='left')
    
    # If no click on mini-map, show top 10 species
    if minimap_click is None:
        
        # Top 10 species based only on the main hexagon
        main_hex_obs_with_plants = hex_obs_with_plants[hex_obs_with_plants['h3_cell'] == h3_cell].copy()
        species_counts = main_hex_obs_with_plants['English name'].value_counts().head(10)
        
        # Handle case where main hexagon has no species
        if species_counts.empty:
            return html.Div([
                html.H5("Top 10 Species", style={'marginTop': '15px', 'marginBottom': '10px'}),
                html.P("No observations in this hexagon.", style={'color': '#6c757d', 'fontStyle': 'italic'})
            ])

        return html.Div([
            html.H5("Top 10 Species", style={'marginTop': '15px', 'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Strong(species, style={'fontSize': '12px'}),
                    html.Span(f" ({count})", 
                             style={'color': '#6c757d', 'marginLeft': '6px', 'fontSize': '11px'})
                ], style={'marginBottom': '6px'})
                for species, count in species_counts.items()
            ], style={'maxHeight': '200px', 'overflowY': 'auto'})
        ])
    
    # Get clicked observation details
    point = minimap_click['points'][0]
    
    if 'customdata' not in point or not point['customdata']:
        return html.Div([
            html.P("Observation data not available", style={'color': '#6c757d', 'fontStyle': 'italic'})
        ])
    
    clicked_id = point['customdata'][0]
    
    # Now use the broader hex_obs_with_plants which includes neighbors
    clicked_obs = hex_obs_with_plants[hex_obs_with_plants['ourID'] == clicked_id] 
    
    if clicked_obs.empty:
        return html.Div([
            html.P("Observation not found", style={'color': '#6c757d', 'fontStyle': 'italic'})
        ])
    
    clicked_obs = clicked_obs.iloc[0]
    
    # Format the date
    obs_date = "Unknown"
    if pd.notnull(clicked_obs['eventDate']):
        obs_date = clicked_obs['eventDate'].strftime('%Y-%m-%d')
    
    category = clicked_obs['Category']
    
    return html.Div([
        # Header and back button
        html.Div([
            html.H5("Observation Details", style={'marginTop': '10px', 'marginBottom': '5px', 'display': 'inline-block'}),
            html.Button("← Back to Top 10", 
                       id={'type': 'back-btn', 'index': 0},
                       n_clicks=0,
                       style={
                           'padding': '4px 8px',
                           'fontSize': '11px',
                           'backgroundColor': '#f8f9fa',
                           'border': '1px solid #dee2e6',
                           'borderRadius': '4px',
                           'cursor': 'pointer',
                           'float': 'right'
                       })
        ], style={'marginBottom': '10px', 'overflow': 'auto'}),
        
        # Info box
        html.Div([
            # English name with icon
            html.Div([
                html.Img(src=f"/assets/icons/{category}.svg", height=28, 
                         style={'verticalAlign': 'middle', 'marginRight': '8px'}),
                html.H4(clicked_obs['English name'], 
                        style={'display': 'inline-block', 'margin': '0', 'verticalAlign': 'middle'})
            ], style={'marginBottom': '8px', 'borderBottom': '1px solid #eee', 'paddingBottom': '8px'}),
            
            # Latin name | Danish name
            html.Div([
                html.Span(clicked_obs['Latin name'], 
                          style={'fontSize': '12px', 'color': '#555', 'fontStyle': 'italic'}),
                html.Span(" | ", style={'margin': '0 8px', 'color': '#ccc'}),
                html.Span(clicked_obs['Danish name'], 
                          style={'fontSize': '12px', 'color': '#555'}),
            ], style={'marginBottom': '15px', 'textAlign': 'left'}),
            
            # Category (now shown as text along with icon)
            html.Div([
                html.Strong("Category: ", style={'fontSize': '13px'}),
                html.Span(category, style={'fontSize': '13px'})
            ], style={'marginBottom': '8px'}),
            
            # Date
            html.Div([
                html.Strong("Date: ", style={'fontSize': '13px'}),
                html.Span(obs_date, style={'fontSize': '13px'})
            ], style={'marginBottom': '8px'}),
            
            # Location
            html.Div([
                html.Strong("Location: ", style={'fontSize': '13px'}),
                html.Span(f"{clicked_obs['decimalLatitude']:.5f}°, {clicked_obs['decimalLongitude']:.5f}°", 
                         style={'fontSize': '13px'})
            ], style={'marginBottom': '8px'}),
            
        ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
    ])


if __name__ == '__main__':
    app.run(debug=True)