import os
from collections import namedtuple

import numpy as np
from scipy.linalg import svd

from .region_grower import RegionGrower
from .peak_integrator import PeakIntegrator

try:
    from single_peak_integration import (
        get_grid_data,
        rotation_matrix,
        gaussian_mixture,
        rebin_histogram,
        marginalize_1d,
        marginalize_2d,
    )
except Exception as e:
    pass


IntegrationResult = namedtuple('IntegrationResult', [
    'peak_intensity',
    'peak_sigma',
    'bg_intensity',
    'bg_density',
    'peak_hull',
    'inner_hull',
    'outer_hull',
    'status',
])


def integrate_peak(
        intensity,
        mask,
        distance_threshold=2.0,
        min_intensity=10.0,
        max_size=7.0,
        box_search_size=5,
        snap_range=5,
        smoothing_window_size=5,
        min_peak_pixels=30,
        min_peak_snr=1.0,
        background_estimate='mean',
        outlier_threshold=2.0,
        return_hulls=True,
        return_status=True,
        export_visualization=False,
        visualize_norm='linear',
        visualize_file='peak.html'
):
    region_grower = RegionGrower(distance_threshold, min_intensity, max_size)
    peak_int = PeakIntegrator(
        region_grower, 
        box_search_size, 
        smoothing_window_size,
        snap_range,
        min_peak_pixels,
        min_peak_snr,
        background_estimate,
        outlier_threshold
    )
    
    center = np.array(intensity.shape, dtype=int).reshape(1, 3) // 2
    result = peak_int.integrate_peaks(
        bank_id=0, 
        intensity=intensity, 
        peak_centers=center, 
        return_hulls=return_hulls, 
        return_headers=False, 
        return_status=True,
        mask=mask
    )

    if return_hulls:
        data, hulls, status = result
    else:
        data, status, hulls = result, [[None], [None], [None], [None]]

    _, _, bg_den, peak_int, bg_int, sigma = data[0]
    
    if export_visualization:
        if visualize_norm == 'log':
            fig = visualize_peak(np.log(1 + events), hulls[0][1])
        else:
            fig = visualize_peak(events, hulls[0][1])
        fig.write_html(visualize_file)

    return IntegrationResult(
        peak_int, 
        sigma, 
        bg_int, 
        bg_den, 
        peak_hull=hulls[0][1], 
        inner_hull=hulls[0][2], 
        outer_hull=hulls[0][3],
        status=status[0].value
    )


def visualize_peak(intensity, hull, intensity_opacity=0.2):
    x, y, z = np.meshgrid(
        np.arange(intensity.shape[0]),
        np.arange(intensity.shape[1]),
        np.arange(intensity.shape[2]),
        indexing='ij'
    )

    data = []

    import plotly.graph_objects as go
    data.append(go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=intensity,
        opacity=0.15,
        surface_count=17
    ))
 
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

