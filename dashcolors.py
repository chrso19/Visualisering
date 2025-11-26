import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# Your color theme
theme = {
    "color1": "#053913",   # black-forest
    "color2": "#942911",   # brandy
    "color3": "#628b48",   # fern
    "color5": "#d58936",   # bronze
        
    "background": "#f7f3e9",   # cream-light
    "text": "#0b0b09",         # onyx
    "header" : "#04250C"   # Dark green
}

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    style={
        'backgroundColor': theme['background'],
        'minHeight': '100vh',
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif'
    },
    children=[
        html.H1(
            'My Dash Application',
            style={
                'color': theme['header'],
                'textAlign': 'center',
                'marginBottom': '30px'
            }
        ),
        
        html.Div([
            html.Label(
                'Select a value:',
                style={'color': theme['text'], 'fontSize': '18px', 'marginBottom': '10px'}
            ),
            dcc.Slider(
                id='value-slider',
                min=0,
                max=100,
                value=50,
                marks={i: str(i) for i in range(0, 101, 25)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'marginBottom': '40px'}),
        
        html.Div([
            dcc.Graph(id='example-graph')
        ])
    ]
)

# Callback to update the graph
@app.callback(
    Output('example-graph', 'figure'),
    Input('value-slider', 'value')
)
def update_graph(slider_value):
    # Sample data
    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    values = [slider_value, slider_value * 0.8, slider_value * 1.2, slider_value * 0.6]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=[theme['color1'], theme['color2'], theme['color5'], theme['color3']]
        )
    ])
    
    # Update layout with theme colors
    fig.update_layout(
        plot_bgcolor=theme['background'],
        paper_bgcolor=theme['background'],
        font={'color': theme['text']},
        title={
            'text': f'Sample Data (Value: {slider_value})',
            'font': {'color': theme['color3'], 'size': 20}
        },
        xaxis={'gridcolor': theme['text'], 'gridwidth': 0.5},
        yaxis={'gridcolor': theme['text'], 'gridwidth': 0.5}
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)