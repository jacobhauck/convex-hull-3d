from collections import namedtuple

from .region_grower import RegionGrower
from .peak_integrator import PeakIntegrator


IntegrationResult = namedtuple('IntegrationResult', [
    'peak_intensity',
    'peak_sigma',
    'bg_intensity',
    'bg_density',
    'peak_hull',
    'inner_hull',
    'outer_hull',
])


def integrate_peak(
        intensity,
        mask,
        distance_threshold=2.0,
        min_intensity=10.0,
        max_size=7.0,
        box_search_size=5,
        smoothing_window_size=5,
        min_peak_pixels=30,
        min_peak_snr=1.0,
        outlier_threshold=2.0,
        return_hulls=True
):
    region_grower = RegionGrower(distance_threshold, min_intensity, max_size)
    peak_int = PeakIntegrator(
        region_grower, 
        box_search_size, 
        smoothing_window_size,
        min_peak_pixels,
        min_peak_snr,
        outlier_threshold
    )
    
    center = np.array(intensity.shape, dtype=int).reshape(1, 3) // 2
    result = peak_int.integrate_peaks(
        0, 
        intensity, 
        center, 
        return_hulls, 
        False, 
        mask
    )

    if return_hulls:
        data, hulls = result
    else:
        data, hulls = result, None

    _, _, bg_den, peak_int, bg_int, sigma = data[0]

    return IntegrationResult(
        peak_int, 
        sigma, 
        bg_int, 
        bg_den, 
        peak_hull, 
        inner_hull, 
        outer_hull
    )


def visualize_peak(intensity, hull):
    x, y, z = np.meshgrid(
        np.arange(intensity.shape[0]),
        np.arange(intensity.shape[1]),
        np.arange(intensity.shape[2]),
        indexing='ij'
    )

    hull_x, hull_y, hull_z = [], [], []
    for simplex in hull.simplices:
        hull_x.extend(hull.points[simplex, 0])
        hull_x.append(None)
        hull_y.extend(hull.points[simplex, 1])
        hull_y.append(None)
        hull_z.extend(hull.points[simplex, 2])
        hull_z.append(None)

    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=intensity,
            opacity=0.15,
            surface_count=17
        ),
        go.Scatter3d(
            x=hull_x,
            y=hull_y,
            z=hull_z,
            mode='lines'
        )
    ])

    return fig
