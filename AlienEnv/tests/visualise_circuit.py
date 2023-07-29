import pandas as pd
import plotly.express as px

df = pd.read_excel("../data/nord_20230709_0303.xlsx")

x_coords = df["x_coords"]
y_coords = df["y_coords"]
z_coords = df["z_coords"]
timestamps = df["timestamps"]
normalizedCarPositions = df["normalizedCarPositions"]
tyres_out = df["tyres_out"]
speeds = df["speeds"]
off_track = df["off_track"]


fig = px.scatter_3d(df, x=x_coords, y=y_coords, z=z_coords, color=speeds)
fig.update_scenes(aspectmode="data")
fig.update_traces(marker_size=5)
fig.show()
