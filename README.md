# 3D Bragg Peak Integration by Clustering and Convex Hull Fitting

This repository contains the 3D implementation of the Bragg peak 
integration algorithm that uses agglomerative clustering followed
by convex hull fitting.

## Installation

Clone this repository, and then use `pip`
```commandline
pip install .
```

Only hard requirements are NumPy and SciPy. Optionally,
you can install plotly for 3D visualization of peaks using
```commandline
pip install ".[vis]"
```

## Usage

The main class is `PeakIntegrator`. Construct a `PeakIntegrator`
object and call `integrate_peaks` to find the peaks in a 3D
event histogram:
```python
import convex_hull

# Configure integration algorithm
integrator = convex_hull.PeakIntegrator(
    min_intensity=15.0,
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

# Load your event array
events = ...  # (D, H, W) array containing histogram of events

# Integrate peaks
peaks = integrator.integrate_peaks(events)
# peaks is a list of IntegrationResult objects
```
This results in a list containing one `IntegrationResult` for a peak
at the center of the histogram. Pass a custom array of starting points to 
integrate multiple peaks in one histogram.

```python
# Set up a custom set of potential peak locations
# (default is one peak at the center of the histogram)
centers = ...  # (N, 3) array of peak center coordinates

# Integrate from custom starting centers
peaks = integrator.integrate_peaks(events, peak_centers=centers)
```

You can also use a custom mask array to indicate which pixels are
valid.

```python
# Load optional detector mask
detector_mask = ...  # (D, H, W) mask

# Integrate peaks accounting for mask
peaks = integrator.integrate_peaks(events, mask=detector_mask)
```

You can visualize the integration result in 3D by using the
`visualize_peak` function.

```python
fig = convex_hull.visualize_peak(events, peaks[0].peak_hull)
fig.show()  # or fig.write_html(), ...
```

### Parameters of `PeakIntegrator`

- `min_intensity`: number, units events/pixel
    - Minimum average intensity (# of events) of neighboring pixels required for a pixel to be included in a peak
    - Set this slightly above the average noise level of each image
    - $\downarrow$ for more and bigger peaks
    - $\uparrow$ to remove false positives and decrease peak size
    - This is the **most important** parameter
- `distance_threshold`: number, units pixels
    - Radius of neighborhood around a pixel in which the average neighbor intensity is calculated (see `min_intensity`)
- `max_size`: number, units pixels
    - Maximum radius in pixels within which pixels can be added to the peak (measured from candidate/predicted peak location)
    - Set this slightly larger than the largest peak size
    - Use this to cap the size of peak regions if you need to set `region-growth-minimum-intensity` relatively low to avoid false negatives
- `box_size`: odd positive integer, units pixels
    - Size of box in which to search for (smoothed) local max pixel from which to start peak clustering
    - Set this around the size of a typical peak
    - Use this to snap peak candidates/predictions to the center of peaks if they are slightly off--predicted locations won't snap farther than `peak-center-box-size`
    - Warning: set this too large, and you run the risk of a predicted peak snapping to stronger, nearby peak.
- `smoothing_window_size`: odd positive integer, units pixels
    - Size of smoothing window in pixels
    - Set ths around the size of a typical peak
    - Not a sensitive parameter; slightly improves performance of peak snapping
- `snap_range`: positive integer, units pixels
    - Range in which to search for a nonzero starting point
    - This allows the algorithm to snap onto a nearby peak in case the given starting point happens to be zero for some reason
    - Not a very sensitive parameter, as it seldom comes into play
- `min_peak_pixels`: number, units number of pixels
  - Peaks are discarded if they don't have at least this many pixels
  - Set this a bit smaller than the smallest peak area
  - $\uparrow$ to filter out small false-positive detections caused by noise
  - $\downarrow$ to avoid false negatives if small peaks are present
- `min_peak_snr`: number, unitless
    - Peaks are discarded if the signal-to-noise ratio of the integrated intensity is below this threshold
    - Setting depends on quality of data
    - $\uparrow$ to filter out false positives not filtered by `peak-minimum-pixels`
    - $\downarrow$ to avoid false negatives in noisy data
- `background_estimate`: string, either `"mean"` or `"median"`
    - Sets the method to use for estimating the background strength in an annular shell around the peak hull
    - `"mean"` uses the mean of the background within the shell
    - `"median"` uses the median of the background within the shell
- `outlier_threshold`: number, unitless
    - Peak pixels will be removed before fitting hull if their intensity is more than `peak-pixel-outlier-threshold` standard deviations away from the average intensity of pixels in the peak, which guards against the effects of noise and improves the quality of the final fitted hull
    - $\uparrow$ less conservative, more irregular peaks
    - $\downarrow$ more conservative, more regular peaks


## Algorithm

Below is a description of the convex hull peak fitting algorithm, including how each parameter affects the algorithm.

1. The algorithm starts with a set of candidate/predicted peak locations $(x_0, y_0, z_0)$ (in pixel coordinates) and an intensity image $I(x,y,z)$
2. Then an averaging box filter with size `smoothing_window_size`--which has to be an odd, positive integer to make the implementation easy for me--is used to smooth $I$ to produce a smoothed image $I_\text{s}$
3. In a box of **total size** (width and height) `box_size` centered on $(x_0,y_0,z_0)$, an adjusted center $(x_1,y_1,z_1)$ is selected by finding the maximum intensity in $I_\text{s}$, the smoothed image. This allows slightly incorrect initial locations $(x_0,y_0,z_0)$ to "snap" onto the peak. Again, this must be an odd, positive integer for simplicity of implementation (even won't cause an error, but internally the odd integer one less will be used)
4. **CRITICAL STEP** Clustering is used to create a set of peak pixels $P$. Start by setting ${P = \lbrace(x_1, y_1, z_1)\rbrace}$--just the adjusted initial point. Then repeat the following until no points are added to $P$:
    - For each point $(x,y,z) \notin P$ such that $d((x,y,z), P) < {}$`distance_threshold`, calculate the average intensity ${A(x,y,z) = \frac{1}{|N|}\sum\limits_{(x_\text{n}, y_\text{n}, z_\text{n})\in N}I(x_\text{n},y_\text{n},z_\text{n})}$ in the neighborhood $N$ of $(x,y,z)$ defined by ${N = \lbrace(x_\text{n}, y_\text{n}, z_\text{n}) : d((x, y, z), (x_\text{n}, y_\text{n}, z_\text{n})) \le r_1,\text{ } d((x_1,y_1,z_1), (x_\text{n},y_\text{n},z_\text{n})) < r_2\rbrace}$, where $r_1 = {}$`distance_threshold` and $r_2={}$`max_size`. If $A(x,y,z) \ge {}$`min_intensity`, then add $(x,y,z)$ to $P$
5. Calculate the mean $\mu$ and standard deviation $\sigma$ of $\lbrace I(x,y,z) : (x,y,z) \in P\rbrace$. Remove points $(x,y,z)$ from $P$ if $|I(x,y,z) - \mu|/\sigma > {}$`outlier_threshold`
6. If $|P| <{}$`min_peak_pixels`, then discard this peak
7. Compute the convex hull $H_\text{core}$ of $P$, which we will call the core peak hull.
8. Expand the core peak hull by scaling by a factor $s$ about its centroid to produce convex hulls $H_\text{peak}$ (the true peak hull, $s=1.1$), $H_\text{inner}$ (noise estimation inner boundary, $s=1.6$) and $H_\text{outer}$ (noise estimation outer boundary, $s=2.6$)
9. Integrate over the true peak hull $H_\text{peak}$ to obtain the peak intensity $I_\text{peak}$ and uncertainty $\sigma_\text{peak}$. Background noise is subtracted by estimating the average background noise per pixel by integrating over the annulus-like region $H_\text{outer} \setminus H_\text{inner}$.
10. If $I_\text{peak} / \sigma_\text{peak} <{}$`min_peak_snr`, then discard this peak
