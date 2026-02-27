"""
A simple example of a perfectly Gaussian-shaped peak
"""
import numpy as np
import plotly.graph_objects as go

import convex_hull
from convex_hull.thresholding import SNRScanThreshold

# ===== Parameters =====
# Number of histogram bins
BINS = 64

# Synthetic peak shape parameter
SIGMA = np.array([
    [40, 6, -8],
    [6, 12, 1.1],
    [-3, 1.1, 5]
])

# Synthetic peak intensity parameters
MAX_INTENSITY = 200
BACKGROUND = 30

# Synthetic detector mask
MASK = np.full((BINS, BINS, BINS), True)
MASK[:, :8, :] = False  # mask out 8 pixels in the y direction

def main():
    # ===== Generate synthetic peak events using a Poisson process =====
    x, y, z = np.indices((BINS, BINS, BINS), dtype=float)
    xyz = np.stack([x, y, z], axis=-1)  # (BINS, BINS, BINS, 3)
    centered = xyz - (BINS/2, BINS/2, BINS/2)
    sigma_inv = np.linalg.inv(SIGMA)
    z2 = np.einsum('xyzi,ij,xyzj->xyz', centered, sigma_inv, centered)
    # (BINS, BINS, BINS)
    rate = BACKGROUND + MAX_INTENSITY * np.exp(-z2 / 2)
    events = np.random.poisson(rate) * MASK

    # ===== Run convex hull peak fitting =====

    # Use BACKGROUND as default min_intensity threshold
    integrator = convex_hull.PeakIntegrator(
        BACKGROUND,
        distance_threshold=3.0,
        max_size=28.0,
        box_size=3,
        smoothing_window_size=3,
        snap_range=8,
        min_peak_pixels=200,
        min_peak_snr=1.0,
        background_estimate='median',
        outlier_threshold=2.0
    )

    # Construct SNR scan threshold calculator
    snr_scan = SNRScanThreshold(
        start_intensity=0.5,
        scan_steps=100,
        integrator=integrator,
        min_threshold=0.5,
        num_processes=4
    )

    # Use SNR scan to find optimal threshold
    min_intensity = snr_scan.get_threshold(events, MASK)
    snr_scan.close()  # Close processing pool (if applicable)

    # Run final convex hull fitting
    integrator.region_grower.min_intensity = min_intensity
    result = integrator.integrate_peaks(events, mask=MASK)[0]

    # Print results
    print(f'Fitted peak using min_intensity = {min_intensity} ')
    print(f'Peak intensity: {result.peak_intensity} '
          f'(theoretical intensity = '
          f'{MAX_INTENSITY * (2 * np.pi) ** (3/2) * np.linalg.det(SIGMA) ** .5})')
    print(f'Intensity sigma: {result.peak_sigma}')
    print(f'Background density: {result.bg_density}')
    print(f'Peak volume: {result.peak_hull.volume}')

    # ===== Visualize fitted peak =====
    fig1 = go.Figure(data=[
        go.Scatter(
            x=snr_scan.scan_min_intensities,
            y=snr_scan.snr_smooth,
            name='Effect of min_intensity on SNR'
        )
    ])
    fig1.update_layout(xaxis_title='min_intensity', yaxis_title='SNR')
    fig1.show()
    fig1.write_html('example_snr_curve.html')

    fig = convex_hull.visualize_peak(events, result.peak_hull, intensity_opacity=0.06)
    fig.show()
    fig.write_html('example_peak.html')


if __name__ == '__main__':
    main()
