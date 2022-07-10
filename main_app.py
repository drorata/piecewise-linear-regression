import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.stats import linregress
import plotly.graph_objects as go
from functools import partial
from numbers import Number
from typing import List


def line_plot(
    x1: Number, x2: Number, slope: Number, intercept: Number, factor: Number
) -> List[go.Scatter]:
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


def window_lin_reg(window: pd.DataFrame, win_len: int) -> List[Number]:
    if window.shape[0] < win_len:
        return [None, None, None, None]

    lin_reg = linregress(window["x"], window["y"])
    return [window["x"].iloc[0], window["x"].iloc[-1], lin_reg.slope, lin_reg.intercept]


"""
# Playing with Piecewise-linear-regression

Made with ⚡️ by Dror Atariah

## Background

In another project I'm working on, I had to experiment with _piecewise-linear-regression_.
The best way to move forward in this case is to experiment.
Here's the result of the experimentation.

## Environment

As I wanted to take advantage of this exercise and deploy the "experiment" using
[Streamlit cloud](https://share.streamlit.io/),
I had to switch from my long standing workflow (using `conda`) to `pyenv` together with `pipenv`.
For more context,
checkout [this thread](https://discuss.streamlit.io/t/failing-to-build-a-streamlit-cloud-app/27494/2?u=drorata).
Switching to the new setup was relatively smooth.
I might stick to it.

## Flow

* Select the min/max ranges of the x-axis
"""

X_MIN = st.number_input(
    label="Min of x-axis", min_value=-10.0, max_value=0.0, value=0.0, step=0.1,
)
X_MAX = st.number_input(
    label="Max of x-axis", min_value=0.0, max_value=10.0, value=2 * np.pi, step=0.1,
)
POINTS_NUM = st.number_input(
    label="Number of points on the x-axis",
    min_value=10,
    max_value=10000,
    value=1000,
    step=10,
)
NOISE = st.number_input(label="Noise level", min_value=0.0, max_value=1.0, value=0.0)

x = np.linspace(X_MIN, X_MAX, num=POINTS_NUM, endpoint=True)
y = np.cos(x) + np.random.uniform(low=-NOISE, high=NOISE, size=x.shape[0])

df = pd.DataFrame({"x": x, "y": y})
fig = px.line(df, x="x", y="y")
st.plotly_chart(fig, use_container_width=True)

"""
At this point, `df` holds a function (currently hardcoded).
See 👆🏻.

Next, you can select the width of the rolling window that will be used to compute the linear regressions.
"""

win_len = st.slider(
    label="Window's width",
    min_value=2,
    max_value=df.shape[0],
    value=int(df.shape[0] / 2),
)
regression_data = list(
    map(partial(window_lin_reg, win_len=win_len), df.rolling(window=int(win_len)))
)

"""
For the sake of nicer visualizations, you might want to skip every n-regression.
"""
SKIP_EVERY_N = st.number_input(
    "Skip every N segment. Select N", min_value=1, max_value=300, value=2
)

"""
We're ready to plot the result.
The colors of the lines are changing as with every new regression.
"""
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

"""
I strongly recommend forking this repo and playing around with various functions.
You can get some nice looking plots! 🤓
"""
