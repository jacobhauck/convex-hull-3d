import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull


def visualize_peak(
        intensity: npt.NDArray,
        hull: ConvexHull | None,
        intensity_opacity=0.15,
        norm: str = 'linear'
):
    """
    Creates a plotly Volume plot of the given peak histogram with optional
    convex hull overlaid.

    Parameters
    ----------
    intensity: (D, H, W) array of intensity values
    hull: ConvexHull object of convex hull to visualize (or None, to visualize
        only the intensity)
    intensity_opacity: Opacity of intensity isosurfaces in volume plot
    norm: Normalization option for intensity. Choose 'linear' for linear color
        scale. Choose 'log' for a logarithmic color scale

    Return
    ------
    plotly figure containing traces for the volume plot and the optional convex
    hull
    """
    import plotly.graph_objects as go
    x, y, z = np.meshgrid(
        np.arange(intensity.shape[0]),
        np.arange(intensity.shape[1]),
        np.arange(intensity.shape[2]),
        indexing='ij'
    )

    if norm == 'log':
        intensity = np.log(1 + intensity)
    elif norm != 'linear':
        raise ValueError('Invalid norm.')

    data = [go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=intensity,
        opacity=intensity_opacity,
        surface_count=17
    )]

    if hull is not None:
        hull_x, hull_y, hull_z = [], [], []
        for simplex in hull.simplices:
            hull_x.extend(hull.points[simplex, 0])
            hull_x.append(None)
            hull_y.extend(hull.points[simplex, 1])
            hull_y.append(None)
            hull_z.extend(hull.points[simplex, 2])
            hull_z.append(None)
        
        data.append(go.Scatter3d(
            x=hull_x,
            y=hull_y,
            z=hull_z,
            mode='lines'
        ))

    fig = go.Figure(data=data)

    return fig
