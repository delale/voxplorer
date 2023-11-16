"""
Embedding projector code.

This is an attempt at imitating the functionalities in TensorBorad's Embedding Projector,
adding some additional functionalities such as dynamic data download based on graphical
selection.
"""
import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from sklearn.decomposition import PCA


# Load high dimensional data
df = pd.read_csv('./testing/data_load_test.csv')
labels = df[['y0', 'y1', 'z']]
df = df.drop(columns=['y1', 'y0', 'z'])

# Perform dim reduction
pca = PCA(n_components=3)
df_reduced = pd.DataFrame(pca.fit_transform(df), columns=['PC1', 'PC2', 'PC3'])
df_reduced['label'] = labels['z']

# Plot the reduced data
fig = px.scatter_3d(df_reduced, x='PC1', y='PC2', z='PC3', color='label')
fig.update_layout(dragmode='select')

# Create the app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='3dplot', figure=fig),
    html.Button('Download', id='download-button'),
    dcc.Link('', id='download-link', href='#')
])


@app.callback(
    Output('download-link', 'href'),
    Input('download-button', 'n_clicks'),
    State('3dplot', 'selectedData')
)
def update_download_button(n_clicks, selectedData):
    if n_clicks is not None and selectedData is not None:
        points = selectedData['points']
        indices = [point['pointIndex'] for point in points]
        df_selected = df_reduced.iloc[indices]
        print(df_selected)


if __name__ == '__main__':
    app.run_server(debug=True)
