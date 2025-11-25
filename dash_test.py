import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc

import os
import pandas as pd
import random
import numpy as np


data = pd.read_pickle("GBIF_data.pkl")

observations = data["observations"]
plants = data["plants"]

del data

key = plants.loc[plants['English name'] == "Wild garlic", 'taxonKey'].values[0]

fig = px.histogram(
    observations[np.logical_and(observations['taxonKey']==key, observations['year']==2024)],
    x='month',     
)

fig.update_layout(
    title = "Wild Garlic observations per month in 2024"
)

key = plants.loc[plants['English name'] == "Garlic mustard", 'taxonKey'].values[0]

fig2 = px.histogram(
    observations[np.logical_and(observations['taxonKey']==key, observations['year']==2024)],
    x='month',     
)

fig2.update_layout(
    title = "Garlic mustard observations per month in 2024"
)

app = dash.Dash(__name__)

app.layout = html.Div(
    children = [
        html.H1(children = "Foraging in Denmark"),
        html.P(
            children = (
                "Presentation of different edible plants and funghi in Denmark"
                " with locations and observations over the past 3 years"
            ),
        ),
        dcc.Graph(figure = fig),
        dcc.Graph(figure = fig2)
    ]
)

if __name__ == "__main__":
    app.run(debug=True)