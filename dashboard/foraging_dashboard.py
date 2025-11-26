import pickle
import dash
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output

# Load data
data = pickle.load(open("GBIF_data.pkl", "rb"))
plants = data["plants"]
observations = data["observations"]

plants_df = plants[["taxonKey", "species", "English name", "Category"]]
obs_df = observations[["taxonKey", "eventDate", "decimalLatitude", "decimalLongitude"]]

# Merge on taxonKey
merged = obs_df.merge(plants_df, on="taxonKey", how="left")

subset = merged.sample(n=200, random_state=42)

# Create scatter map
fig = px.scatter_map(
    subset,
    lat="decimalLatitude",
    lon="decimalLongitude",
    hover_name="English name",
    hover_data=["English name", "Category"],
    zoom=5,
    map_style="open-street-map",
    width=1400,  
    height=800
)

categories = ["herb", "berry", "mushroom", "flower", "apple"]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Foraging in Denmark", style={"textAlign": "center"}),
    html.Div([
        # Checklist on the left
        html.Div([
            dcc.Checklist(
                id="category-selector",
                options=[
                    {
                        "label": [
                            html.Img(src=f"/assets/images/{cat}.png", height=30),
                            html.Span(cat.capitalize(), style={"font-size": 15, "padding-left": 10})
                        ],
                        "value": cat
                    }
                    for cat in categories
                ],
                labelStyle={"display": "flex", "align-items": "center", "margin-bottom": "10px"},
                value=categories
            )
        ], style={"width": "10%", "padding": "10px"}),  # checklist takes 20% of width

        # Map on the right
        html.Div([
            dcc.Graph(id="scatter-map")
        ], style={"width": "90%"})  # map takes 80% of width
    ], style={"display": "flex"})
])

@app.callback(
    Output("scatter-map", "figure"),
    Input("category-selector", "value")
)
def update_map(selected_categories):
    # Filter data based on selected categories
    filtered = subset[subset["Category"].isin(selected_categories)]

    # Create map
    fig = px.scatter_map(
        subset,
        lat="decimalLatitude",
        lon="decimalLongitude",
        hover_name="English name",
        hover_data=["English name", "Category"],
        zoom=5,
        map_style="open-street-map",
        width=1400,  
        height=800
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
