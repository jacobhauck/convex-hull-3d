from convex_hull.peak_integrator import PeakIntegrator
from convex_hull.region_grower import RegionGrower
import numpy as np
import plotly.graph_objects as go


peak_integrator = PeakIntegrator(
    RegionGrower(
        distance_threshold=1.8,
        min_intensity=3.0,
        max_size=7.5
    ),
    box_size=3,
    smoothing_window_size=3,
    min_peak_pixels=10
)

x_center, y_center, z_center = (25, 25, 25)
x_sigma, y_sigma, z_sigma = (2.5, 2.5, 2.5)
background = 2
peak = 30
integral = (2*np.pi) ** (3/2) * x_sigma * y_sigma * z_sigma * peak

x, y, z = np.meshgrid(
    np.arange(50),
    np.arange(50),
    np.arange(50),
    indexing='ij'
)

rate = background + peak * np.exp(-(((x - x_center)/x_sigma)**2 + ((y - y_center)/y_sigma)**2 + ((z - z_center)/z_sigma)**2) / 2)
rng = np.random.default_rng()
intensity = rng.poisson(rate)
integral_num = rate.sum() - 2 * rate.flatten().shape[0]

centers = np.array([[26, 24, 25]])
data, hulls = peak_integrator.integrate_peaks(0, intensity, centers, return_headers=True, return_hulls=True)
print("True integral?", integral)
print(data)
print(hulls)
hull_x = []
hull_y = []
hull_z = []
for simplex in hulls[0][1].simplices:
    hull_x.extend(hulls[0][1].points[simplex, 0])
    hull_x.append(None)
    hull_y.extend(hulls[0][1].points[simplex, 1])
    hull_y.append(None)
    hull_z.extend(hulls[0][1].points[simplex, 2])
    hull_z.append(None)


fig = go.Figure(data=[
    go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=intensity.flatten(),
        opacity=0.04,
        surface_count=17
    ),
    go.Scatter3d(
        x=hull_x,
        y=hull_y,
        z=hull_z,
        mode='lines'
    )
])
fig.show()
