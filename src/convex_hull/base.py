import os
from collections import namedtuple

import numpy as np
from scipy.linalg import svd

from .region_grower import RegionGrower
from .peak_integrator import PeakIntegrator
from single_peak_integration import (
    get_grid_data,
    rotation_matrix,
    gaussian_mixture,
    rebin_histogram,
    marginalize_1d,
    marginalize_2d,
)


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
        data, hulls = result, [None, None, None, None]

    _, _, bg_den, peak_int, bg_int, sigma = data[0]

    return IntegrationResult(
        peak_int, 
        sigma, 
        bg_int, 
        bg_den, 
        peak_hull=hulls[0][1], 
        inner_hull=hulls[0][2], 
        outer_hull=hulls[0][3]
    )


def plot_fit(
        hist_ws,
        gauss_params,
        bins,
        integration_result,
        plot_path='output',
        prefix=None,
        peak_id=None,
        peak_hkl=None,
        peak_std=4,
        bkgr_std=10,
        detector_mask=None,
        log=False
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    # create output directory for plots
    os.makedirs(plot_path, exist_ok=True)

    # fit model to data
    data, points, edges = get_grid_data(hist_ws, return_edges=True)

    # point along each dimension
    dim_points = [points[0][:,0,0],points[1][0,:,0],points[2][0,0,:]]

    # parameters of the model
    bkgr     = gauss_params[0]**2
    intst    = gauss_params[1]**2
    mu       = gauss_params[2:5]
    angles   = gauss_params[5:8]
    sqrt_eig = 1 / np.array(gauss_params[8:11])
    num_peak_params = (len(gauss_params)-1)//2

    # inverse rotation matrix
    R = rotation_matrix(angles)

    # full covariance matrix
    cov_3d = R.T @ np.diag(sqrt_eig**2) @ R

    # covariances and ellipsoids of 2d marginals
    cov_2d   = []
    angle_2d = []
    sigma_2d = []
    for i in range(3):
        yind, xind = [j for j in range(3) if j!=i]
        cov_2d.append( cov_3d[np.ix_([xind,yind],[xind,yind])] )
        roti, eigi, _ = svd(cov_2d[-1])
        angle_2d.append( np.sign(roti[0,1]) * np.arccos(roti[0,0])/np.pi*180 )
        sigma_2d.append( np.sqrt(eigi) )

    # fitted model
    fit = bkgr + gaussian_mixture(gauss_params[1:], points.reshape((3,-1)), covariance_parameterization='givens').reshape(data.shape)
    fit_masked = (1 if detector_mask is None else detector_mask) * fit

    # rebinned data and fit
    rebinned_data, rebinned_points, rebinned_edges = get_grid_data(hist_ws, bins, rebin_mode='density', return_edges=True)
    rebinned_fit = rebin_histogram(fit, bins, mode='density')


    ########################################
    # Initialize plot
    fig = plt.figure(constrained_layout=True, figsize=(20, 45))
    axes = fig.subplots(8, 3)
    ax_id = -1
    
    ########################################
    # plot 1d marginals        
    ax_id += 1
    normalize = False
    data_1d = marginalize_1d(data, normalize=normalize, detector_mask=detector_mask)
    fit_1d  = marginalize_1d(fit,  normalize=normalize, detector_mask=detector_mask)
    rebinned_data_1d = marginalize_1d(rebinned_data, normalize=normalize, bins=bins, detector_mask=detector_mask)
    rebinned_fit_1d  = marginalize_1d(rebinned_fit,  normalize=normalize, bins=bins, detector_mask=detector_mask)
   
    for i,ax in enumerate(axes[ax_id]):
        ax.stairs(data_1d[i], edges=edges[i], fill=True)
        ax.stairs(rebinned_data_1d[i], edges=rebinned_edges[i], fill=False, lw=1.5)
        ax.plot(dim_points[i], fit_1d[i])
        ax.vlines([mu[i]-peak_std*np.sqrt(cov_3d[i,i]),mu[i]+peak_std*np.sqrt(cov_3d[i,i])], 0, data_1d[i].max(), color='r', ls='-')
        ax.vlines([mu[i]-bkgr_std*np.sqrt(cov_3d[i,i]),mu[i]+bkgr_std*np.sqrt(cov_3d[i,i])], 0, data_1d[i].max(), color='r', ls='-.')
        ax.set_xlabel(hist_ws.getDimension(i).name, fontsize='x-large')
        if i == 0:
            ax.legend(['data','reb. data','fit', f'{peak_std} sigma', f'{bkgr_std} sigma'], framealpha=1.0, fontsize='xx-large')
        ax.set_box_aspect(1)
    
    ########################################
    # Plot rebinned 1D marginals
    ax_id += 1
    for i,ax in enumerate(axes[ax_id]):
        ax.stairs(rebinned_data_1d[i], edges=rebinned_edges[i], fill=True)
        ax.stairs(rebinned_fit_1d[i],  edges=rebinned_edges[i], fill=False, lw=1.5)
        ax.plot(dim_points[i], fit_1d[i])
        ax.set_xlabel(hist_ws.getDimension(i).name, fontsize='x-large')
        if i==0:
            ax.legend(['reb. data','reb. fit', 'fit'], fontsize='xx-large')
        ax.set_box_aspect(1)

    ########################################
    # plot 2d marginals
    data_2d = marginalize_2d(data, normalize=normalize)
    fit_2d  = marginalize_2d(fit,  normalize=normalize)
    fit_masked_2d  = marginalize_2d(fit_masked,  normalize=normalize)
    rebinned_data_2d = marginalize_2d(rebinned_data, normalize=normalize, bins=bins, original_shape=True)
    rebinned_fit_2d  = marginalize_2d(rebinned_fit,  normalize=normalize, bins=bins, original_shape=True)

    # show zero pixels as None
    data_2d = [d/(d!=0) for d in data_2d]
    fit_2d  = [d/(d!=0) for d in fit_2d]
    fit_masked_2d = [d/(d!=0) for d in fit_masked_2d]
    rebinned_data_2d = [d/(d!=0) for d in rebinned_data_2d]
    rebinned_fit_2d  = [d/(d!=0) for d in rebinned_fit_2d]

    if log:
        data_2d = np.log(data_2d)
        fit_2d = np.log(fit_2d)
        fit_masked_2d = np.log(fit_masked_2d)
        rebinned_data_2d = np.log(rebinned_data_2d)
        rebinned_fit_2d = np.log(rebinned_fit_2d)

    ########################################
    # original data
    ax_id += 1
    for i,ax in enumerate(axes[ax_id]):
        yind, xind = [j for j in range(3) if j!=i]
        left, right, bottom, top = edges[xind][0], edges[xind][-1], edges[yind][0], edges[yind][-1]
        ax.imshow(data_2d[i], interpolation='none', extent=(left,right,bottom,top), origin='lower')
        ax.add_patch(Ellipse((mu[xind],mu[yind]), 2*peak_std*sigma_2d[i][0], 2*peak_std*sigma_2d[i][1], angle=angle_2d[i], color='red', ls='-', fill=False))
        ax.add_patch(Ellipse((mu[xind],mu[yind]), 2*bkgr_std*sigma_2d[i][0], 2*bkgr_std*sigma_2d[i][1], angle=angle_2d[i], color='red', ls='-.', fill=False))
        if i==0:
            ax.legend([f'{peak_std} sigma',f'{bkgr_std} sigma'], framealpha=1.0, fontsize='xx-large')
        ax.set_xlabel(hist_ws.getDimension(xind).name, fontsize='x-large')
        ax.set_ylabel(hist_ws.getDimension(yind).name, fontsize='x-large')
    
    ########################################
    # gaussian fit
    ax_id += 1
    for i,ax in enumerate(axes[ax_id]):
        yind, xind = [j for j in range(3) if j!=i]
        left, right, bottom, top = edges[xind][0], edges[xind][-1], edges[yind][0], edges[yind][-1]
        ax.imshow(fit_2d[i], interpolation='none', extent=(left,right,bottom,top), origin='lower')
        ax.add_patch(Ellipse((mu[xind],mu[yind]), 2*peak_std*sigma_2d[i][0], 2*peak_std*sigma_2d[i][1], angle=angle_2d[i], color='red', ls='-', fill=False))
        ax.add_patch(Ellipse((mu[xind],mu[yind]), 2*bkgr_std*sigma_2d[i][0], 2*bkgr_std*sigma_2d[i][1], angle=angle_2d[i], color='red', ls='-.', fill=False))
        if i==0:
            ax.legend([f'{peak_std} sigma',f'{bkgr_std} sigma'], framealpha=1.0, fontsize='xx-large')
        ax.set_xlabel(hist_ws.getDimension(xind).name, fontsize='x-large')
        ax.set_ylabel(hist_ws.getDimension(yind).name, fontsize='x-large')

    ########################################
    # Plot rebinned 2D marginals
    ax_id += 1
    for i,ax in enumerate(axes[ax_id]):
        yind, xind = [j for j in range(3) if j!=i]
        left, right, bottom, top = edges[xind][0], edges[xind][-1], edges[yind][0], edges[yind][-1]
        ax.imshow(rebinned_data_2d[i], interpolation='none', extent=(left,right,bottom,top), origin='lower')
        ax.add_patch(Ellipse((mu[xind],mu[yind]), 2*peak_std*sigma_2d[i][0], 2*peak_std*sigma_2d[i][1], angle=angle_2d[i], color='red', ls='-',  fill=False))
        ax.add_patch(Ellipse((mu[xind],mu[yind]), 2*bkgr_std*sigma_2d[i][0], 2*bkgr_std*sigma_2d[i][1], angle=angle_2d[i], color='red', ls='-.', fill=False))
        if i==0:
            ax.legend([f'{peak_std} sigma',f'{bkgr_std} sigma'], framealpha=1.0, fontsize='xx-large')
        ax.set_xlabel(hist_ws.getDimension(xind).name, fontsize='x-large')
        ax.set_ylabel(hist_ws.getDimension(yind).name, fontsize='x-large')

    ######################################## 
    # fit to rebinned data
    ax_id += 1
    for i,ax in enumerate(axes[ax_id]):
        yind, xind = [j for j in range(3) if j!=i]
        left, right, bottom, top = edges[xind][0], edges[xind][-1], edges[yind][0], edges[yind][-1]
        ax.imshow(rebinned_fit_2d[i], interpolation='none', extent=(left,right,bottom,top), origin='lower')
        ax.add_patch(Ellipse((mu[xind],mu[yind]), 2*peak_std*sigma_2d[i][0], 2*peak_std*sigma_2d[i][1], angle=angle_2d[i], color='red', ls='-',  fill=False))
        ax.add_patch(Ellipse((mu[xind],mu[yind]), 2*bkgr_std*sigma_2d[i][0], 2*bkgr_std*sigma_2d[i][1], angle=angle_2d[i], color='red', ls='-.', fill=False))
        if i==0:
            ax.legend([f'{peak_std} sigma',f'{bkgr_std} sigma'], framealpha=1.0, fontsize='xx-large')
        ax.set_xlabel(hist_ws.getDimension(xind).name, fontsize='x-large')
        ax.set_ylabel(hist_ws.getDimension(yind).name, fontsize='x-large')

    ########################################
    # plot difference
    ax_id += 1
    for i,ax in enumerate(axes[ax_id]):
        yind, xind = [j for j in range(3) if j!=i]
        left, right, bottom, top = edges[xind][0], edges[xind][-1], edges[yind][0], edges[yind][-1]
        ax.imshow(np.abs(fit_masked_2d[i]-data_2d[i]), interpolation='none', extent=(left,right,bottom,top), origin='lower')
        ax.set_xlabel(hist_ws.getDimension(xind).name, fontsize='x-large')
        ax.set_ylabel(hist_ws.getDimension(yind).name, fontsize='x-large')
    
    ########################################
    # plot convex hull fit
    if integration_result.peak_intensity is not None:
        ax_id += 1
        for i, ax in enumerate(axes[ax_id]):
            yind, xind = [j for j in range(3) if j != i]
            left, right, bottom, top = edges[xind][0], edges[xind][-1], edges[yind][0], edges[yind][-1]
            ax.imshow(data_2d[i], interpolation='none', origin='lower')
            ax.set_xlabel(hist_ws.getDimension(xind).name, fontsize='x-large')
            ax.set_ylabel(hist_ws.getDimension(yind).name, fontsize='x-large')
            hull = project_hull(integration_result.peak_hull, xind, yind)
            for simplex in hull.simplices:
                ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], c='red')
        
    # save
    if prefix is None:
        plt.savefig(f'{plot_path}/{peak_id}.png')
    else:
        plt.savefig(f'{plot_path}/{prefix}_number_{peak_id}.png')

    plt.close('all')


def project_hull(hull, xind, yind):
    from scipy.spatial import ConvexHull
    projected_points = hull.points[hull.vertices, :][:, [xind, yind]]
    projected_hull = ConvexHull(projected_points)
    return projected_hull


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
