import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.stats import linregress
import plotly.graph_objects as go


def line_plot(x1, x2, slope, intercept, factor):
    y1 = x1 * slope + intercept
    y2 = x2 * slope + intercept
    line_seg = go.Scatter(
        x=(x1, x2),
        y=(y1, y2),
        mode="lines",
        line={"color": f"rgba({200*factor},40,{200*(1-factor)},0.3)"},
    )
    start_pt = go.Scatter(
        x=(x1,), y=(y1,), mode="markers", marker={"color": "green", "opacity": 0.2}
    )
    end_pt = go.Scatter(
        x=(x2,), y=(y2,), mode="markers", marker={"color": "red", "opacity": 0.2}
    )
    return [line_seg, start_pt, end_pt]


"""
# Playing with Piecewise-linear-regression

Made with ⚡️ by Dror Atariah
"""

POINTS_NUM = st.number_input(
    label="Number of points on the x-axis",
    min_value=10,
    max_value=10000,
    value=1000,
    step=10,
)
NOISE = st.number_input(label="Noise level", min_value=0.0, max_value=1.0, value=0.0)

x = np.linspace(0, 10, num=POINTS_NUM, endpoint=True)
y = np.cos(x) + np.random.uniform(low=-NOISE, high=NOISE, size=x.shape[0])

df = pd.DataFrame({"x": x, "y": y})
fig = px.line(df, x="x", y="y")
st.plotly_chart(fig, use_container_width=True)


win_len = st.slider(
    label="Window's width",
    min_value=2,
    max_value=df.shape[0],
    value=int(df.shape[0] / 2),
)
regression_data = []
for window in df.rolling(window=int(win_len)):
    if window.shape[0] < win_len:
        regression_data.append([None, None, None, None])
        continue

    lin_reg = linregress(window["x"], window["y"])
    regression_data.append(
        [window["x"].iloc[0], window["x"].iloc[-1], lin_reg.slope, lin_reg.intercept]
    )


SKIP_EVERY_N = st.number_input(
    "Skip every N segment. Select N", min_value=1, max_value=300, value=2
)
fig = px.line(df, x="x", y="y")
for i, segment in enumerate(regression_data):
    if i % SKIP_EVERY_N == 0:
        factor = i / len(regression_data)
        try:
            traces = line_plot(segment[0], segment[1], segment[2], segment[3], factor)
            for trace in traces:
                fig.add_trace(trace)
        except TypeError:
            pass
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=False)
