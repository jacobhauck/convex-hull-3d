import time

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import svd
from numpy.linalg import eig
# from scipy.interpolate import UnivariateSpline
from scipy.special import loggamma, erf
# from scipy.stats import rv_histogram, rv_discrete
from scipy.ndimage import convolve

import matplotlib
# non-interactive backend that can only write to files
matplotlib.use('agg')
import matplotlib.pyplot as plt



from mantid.simpleapi import *
from mantid.api import *
from mantid.kernel import V3D
from mantid import config
config['Q.convention'] = "Crystallography"



_debug = False



def prod(seq):
	'''Compute the product of a sequence, similar to sum()'''
	p = 1
	for s in seq: p = p * s
	return p

# def expand_dims(a, axis):
# 	for i,ax in enumerate(axis):
# 		a = np.expand_dims(a, axis=ax)
# 	return a
# # # smoke test
# # a = np.ones((5,5))
# # b = expand_dims(a,axis=[1,2])
# # print(b.shape)
# # exit()



def bins_volume(bins):
	vol = 1
	for ax in range(len(bins)):
		# remaining axes
		all_but_ax = tuple(i for i in range(len(bins)) if i!=ax)
		# convert to ndarray to simplify operations
		bins_ax = np.array(bins[ax])
		vol = vol * np.expand_dims(bins_ax, axis=all_but_ax)
	return vol


def rebin_histogram(a, bins=None, inplace=False, mode='mass'):
	r""" Rebin array given bins in each dimension
	For example, if `a = np.ones((5,9))` and `bins = [[2,1,2],[3,2,4]]` then

	with mode=='mass'
		/-----------\
		|111|11|1111|		/-----\
		|111|11|1111|		|6|4|8|
		|---|--|----|		|-|-|-|
	a =	|111|11|1111|	->	|3|2|4|
		|---|--|----|		|-|-|-|
		|111|11|1111|		|6|4|8|
		|111|11|1111|		\-----/
		\-----------/

	with mode=='density'
		/-----------\
		|111|11|1111|		/-----\
		|111|11|1111|		|1|1|1|
		|---|--|----|		|-|-|-|
	a =	|111|11|1111|	->	|1|1|1|
		|---|--|----|		|-|-|-|
		|111|11|1111|		|1|1|1|
		|111|11|1111|		\-----/
		\-----------/

	Inputs
	------
	  a:		n-dimensional array
	  bins:		list with ndim elements, i-th element is a list with bins in the i-th dimension
	  inplace:	reuse array memory or allocate new array
	  mode:		rebin histogram mode, one of [`density`,`mass`]

	Outputs
	-------
	  rebinned array
	"""
	if not inplace: a = a.copy()
	if bins is None:
		return a
	assert a.ndim==len(bins), f"`bins` must have the same length as `a` has dimensions, got a.ndim={a.ndim} and len(bins)={len(bins)}"
	for ax in range(a.ndim):
		assert a.shape[ax]==sum(bins[ax]), f"total number of bins in each dimension must be equal to the corresponding shape of the array, got a.shape[{ax}]=={a.shape[ax]} and sum(bins[{ax}])={sum(bins[ax])}"
		# remaining axes
		all_but_ax = tuple(i for i in range(a.ndim) if i!=ax)
		# convert to ndarray to simplify operations
		bins_ax = np.array(bins[ax])
		# slices of array along the current axis
		slc1 = [slice(None)]*a.ndim;  slc1[ax] = slice(0,len(bins[ax]));    slc1 = tuple(slc1)
		slc2 = [slice(None)]*a.ndim;  slc2[ax] = slice(1,len(bins[ax]));    slc2 = tuple(slc2)
		slc3 = [slice(None)]*a.ndim;  slc3[ax] = slice(0,len(bins[ax])-1);  slc3 = tuple(slc3)
		# note the `out` argument, all operations are in-place
		a = np.cumsum(a, axis=ax, out=a).take(np.cumsum(bins_ax)-1, axis=ax, out=a[slc1])
		# use this instead of np.diff to have in-place difference
		a[slc2] = a[slc2] - a[slc3]
		if mode=='density':
			a[slc1] /= np.expand_dims(bins_ax, axis=all_but_ax) + np.finfo(float).eps
	return np.ascontiguousarray(a)

def split_bins(bins, split, recursive=True):
	""" Split each element of `bins` into `split` equal elements (until it's not possible) """
	if recursive:
		out = [bins]
		while True:
			out1 = split_bins(out[-1], split, False)
			if len(out1)==len(out[-1]):
				return out
			else:
				out.append(out1)
	else:
		# out = [ bi for b in bins for bi in [b//split+1]*(b%split)+[b//split]*(split-b%split) if bi>0 ]
		out = [ bi for b in bins for bi in [b//split+1]*((b%split)//2)+[b//split]*(split-b%split)+[b//split+1]*(b%split-(b%split)//2) if bi>0 ]
	return out

def knuth_bins(a, min_bins=1, max_bins=None, spread=1):
	""" Compute optimal number of bins along each dimension of the histogram `a`.
		Kevin H. Knuth. Optimal data-based binning for histograms and histogram-based probability density models.
		Digital Signal Processing, 95:102581, 2019.

	Inputs
	------
	  a:		ndarray
	  min_bins:	smallest number of bins along each dimension
	  max_bins:	optional, largest number of bins along each dimension
	  spread:	max difference (in powers of 2) in bin sizes of different dimensions

	Outputs
	-------
	  best_bins: optimal bins
	"""

	# shape must be power of 2
	if any([np.log2(s)%1!=0 for s in a.shape]):
		raise ValueError(f"input array must have 2^N_i elements along i-th dimension, got a.shape={a.shape}")

	# total number of events
	N = a.sum()

	# number of dimensions
	ndim = a.ndim

	# diadic hierarchy of histograms
	splits = [ split_bins([s],2,recursive=True) for s in a.shape ]
	min_split = int(np.ceil(np.log2(min_bins)))

	# log likelihoods of each histogram
	logp = np.full([len(s) for s in splits], np.nan)
	it = np.nditer(logp, flags=['multi_index'], op_flags=['readwrite'])
	# with np.nditer(logp, flags=['multi_index'], op_flags=['readwrite']) as it:
	for logpi in it:
		# restrict search space
		if np.abs(np.subtract.outer(it.multi_index,it.multi_index)).max()>spread:
			continue
		# bins along each dimension
		bins = [ split[ind] for split,ind in zip(splits,it.multi_index) ]
		# total number of bins in the rebinned histogram
		M = prod([len(bini) for bini in bins])
		# compute log likelihood of the current histogram
		logpi[...] = N*np.log(M) + loggamma(M/2) - M*loggamma(0.5) - loggamma(N+M/2) + loggamma(rebin_histogram(a,bins,inplace=False,mode='mass')+0.5).sum()

	# rebinning with maximum log likelihood
	best_ind  = np.unravel_index([np.nanargmax(logp)], logp.shape)
	best_ind  = [ max(min_split,i[0]) for i in best_ind]
	best_bins = [ splits[dim][best_ind[dim]] for dim in range(len(splits)) ]
	return best_bins

###############################################################################

def marginalize_1d(arr, mask_missing=False, normalize=False, bins=None, detector_mask=None):
	# number of dimensions
	ndims = arr.ndim

	if detector_mask is not None:
		arr = detector_mask * arr

	out = []
	if bins is not None:
		# find normalization constant
		if normalize:
			dx = prod([np.expand_dims(bins[ax], axis=tuple(i for i in range(ndims) if i!=ax)) for ax in range(ndims)])
			# dx = prod([expand_dims(bins[ax], axis=tuple(i for i in range(ndims) if i!=ax)) for ax in range(ndims)])
			norm_const = (arr*dx).sum()

		# evaluate marginals
		for ax in range(ndims):
			all_but_ax = tuple(i for i in range(ndims) if i!=ax)
			dx = prod([np.expand_dims(bins[ax1], axis=tuple(i for i in range(ndims) if i!=ax1)) for ax1 in all_but_ax])
			# dx = prod([expand_dims(bins[ax1], axis=tuple(i for i in range(ndims) if i!=ax1)) for ax1 in all_but_ax])
			out.append((arr*dx).sum(axis=all_but_ax))

		# normalize marginals
		if normalize:
			out = [a/norm_const for a in out]
		return out

	if normalize:
		arr = arr / arr.sum()
	if mask_missing:
		mask = arr!=0
		return arr.sum(axis=(1,2))/mask.sum(axis=(1,2)), arr.sum(axis=(0,2))/mask.sum(axis=(0,2)), arr.sum(axis=(0,1))/mask.sum(axis=(0,1))
	else:
		out = []
		for ax in range(ndims):
			all_but_ax = tuple(i for i in range(ndims) if i!=ax)
			out.append( arr.sum(axis=all_but_ax) )
		return out
		# return arr.sum(axis=(1,2)), arr.sum(axis=(0,2)), arr.sum(axis=(0,1))


def marginalize_2d(arr, mask_missing=False, normalize=False, bins=None, original_shape=False):
	# number of dimensions
	ndims = arr.ndim

	out = []
	if bins is not None:
		if original_shape:
			for ax in range(ndims):
				arr = np.repeat(arr, bins[ax], axis=ax)
		else:
			# find normalization constant
			if normalize:
				dx = prod([np.expand_dims(bins[ax], axis=tuple(i for i in range(ndims) if i!=ax)) for ax in range(ndims)])
				# dx = prod([expand_dims(bins[ax], axis=tuple(i for i in range(ndims) if i!=ax)) for ax in range(ndims)])
				norm_const = (arr*dx).sum()

			# evaluate marginals
			for ax in range(ndims):#-1,-1,-1):
				all_but_ax = tuple(i for i in range(ndims) if i!=ax)
				dx = np.expand_dims(bins[ax], axis=tuple(i for i in range(ndims) if i!=ax))
				# dx = expand_dims(bins[ax], axis=tuple(i for i in range(ndims) if i!=ax))
				out.append((arr*dx).sum(axis=ax))

			# normalize marginals
			if normalize:
				out = [a/norm_const for a in out]
			return out

	if normalize:
		arr = arr / arr.sum()
	if mask_missing:
		mask = arr!=0
		return arr.sum(axis=0)/mask.sum(axis=0), arr.sum(axis=1)/mask.sum(axis=1), arr.sum(axis=2)/mask.sum(axis=2)
	else:
		return arr.sum(axis=0), arr.sum(axis=1), arr.sum(axis=2)



def find_boundaries(mask):
	bndr = np.zeros_like(mask, dtype=bool)
	bndr[1:,1:] = np.diff(mask>0,axis=0)[:,1:] + np.diff(mask>0,axis=1)[1:,:]
	return bndr

###############################################################################


def eigen(matrix):
	''' Eigendecomposition of the 2x2 symmetric matrix
	    Deledalle, Charles-Alban, et al. Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices. Diss. Université de Lyon, 2017.
	'''
	matrix = np.array(matrix)
	if matrix.ndim>2:
		raise ValueError(f"Input array must be 2 dimensional matrix, got shape={matrix.shape}")
	if matrix.shape[0]!=matrix.shape[1]:
		raise ValueError(f"Input array must be square matrix, got shape={matrix.shape}")
	if (matrix-matrix.T).sum()>1.e-10:
		raise ValueError(f"Input array must be symmetric matrix")

	if matrix.shape[0]==2:
		a,b,c = matrix[0,0], matrix[1,1], matrix[1,0]
		delta = np.sqrt(4*c**2+(a-b)**2)
		# eigenvalues
		v = np.array([(a+b-delta)/2, (a+b+delta)/2])
		# eigenvectors
		if c!=0:
			w = np.array([[(v[1]-b)/c,(v[0]-b)/c],[1,1]])
			w[:,0] = w[:,0] / np.sqrt(np.sum(w[:,0]**2))
			w[:,1] = w[:,1] / np.sqrt(np.sum(w[:,1]**2))
		else:
			w = np.eye(2)
	else:
		raise ValueError("Only 2x2 matrices are supported")

	return v, w



def rotation_matrix(angles):
	''' Compute rotation matrix given angles
		https://en.wikipedia.org/wiki/Givens_rotation#Table_of_composed_rotations
	'''
	# number of dimensions
	if len(angles)==0:
		ndims = 1
	elif len(angles)==1:
		ndims = 2
	elif len(angles)==3:
		ndims = 3
	if ndims>3:
		raise NotImplementedError(f"rotation matrix can be computed only for up to 3 dimensions at the moment, got `ndims={len(angles)}`")
	# cosines and sines of the angles
	c = np.cos(angles)
	s = np.sin(angles)
	# Rotation matrix of 0-1-2 rotation
	if ndims==1:
		R = np.array([[1]])
	elif ndims==2:
		R = np.array([c,-s],[s,c])
	elif ndims==3:
		R = np.array([ [c[1]*c[0],-c[1]*s[0],s[1]], [s[2]*s[1]*c[0]+c[2]*s[0],-s[2]*s[1]*s[0]+c[2]*c[0],-s[2]*c[1]], [-c[2]*s[1]*c[0]+s[2]*s[0],c[2]*s[1]*s[0]+s[2]*c[0],c[2]*c[1]] ])
	return R


def squared_mahalanobis_distance(mu, sqrtP, x):
	r""" Compute squared Mahalanobis distance of the point cloud in `x` as
		d^2(x) = || sqrt(P) @ (x-mu) ||^2 = (x-mu)^T @ P @ (x-mu)

	Inputs
	------
	  mu:		(ndims,) ndarray, coordinates of the peak center
	  sqrtP:	(ndims,ndims) ndarray, square root of the precision matrix
	  x:		(ndims,npoints) ndarray, evaluation points

	Outputs
	-------
	  (npoints,) ndarray, squared mahalanobis distances for each point in x
	"""

	# number of dimensions
	ndims = mu.size

	sqrtP = sqrtP.reshape((ndims,ndims))
	mu    = mu.reshape((ndims,1))

	Px = sqrtP@(x-mu) if ndims>1 else sqrtP[0,0]*(x-mu[0])
	return (Px*Px).sum(axis=0)


def mahalanobis_distance(mu, sqrtP, x):
	r""" Compute Mahalanobis distance of the point cloud in `x` as
		d(x) = \|\sqrt{P} (x-mu)\| = \sqrt{(x-\mu)^T P (x-mu)}

	Inputs
	------
	  mu:		(ndims,) ndarray, coordinates of the peak center
	  sqrtP:	(ndims,ndims) ndarray, square root of the precision matrix
	  x:		(ndims,npoints) ndarray, evaluation points

	Outputs
	-------
	  (npoints,) ndarray, squared mahalanobis distances for each point in x
	"""
	return np.sqrt(squared_mahalanobis_distance(mu, sqrtP, x))


def gaussian(params, x):
	r""" Computes 3D Gaussian function at points in `x`. Gaussian is parameterized as follows
		`params[0]` is the uniform background intensity
		`params[1]` is the max peak intensity
		`params[2:5]` are the x,y,z coordinates of the peak center
		`params[5:14]` are the 9 elements of the flattened square root precision matrix

	Inputs
	------
	  params:	(14,) ndarray, parameters of the Gaussian
	  x:		(3,npoints) ndarray, evaluation points

	Outputs
	-------
	  (npoints,) ndarray, values of the Gaussian at `x`
	"""
	bkgr  = params[0]
	intst = params[1]
	cnt   = params[2:5]
	sqrtP = params[5:]
	return bkgr + intst * np.exp(-0.5*squared_mahalanobis_distance(cnt, sqrtP, x))


def gaussian_mixture(params, x, npeaks=1, covariance_parameterization='givens'):
	r""" Compute nD Gaussian mixture at points in `x`. Parameters are interpreted as follows
		- first `npeaks` values are the max peak intensities of each peak
		- next `npeaks*ndims` values are the coordinates of each of the peak centers
		- remaining values are the parameters of the precision matrix for each of the peaks

	Inputs
	------
	  params:	(1+(1+ndims+ncov_params)*npeaks,) ndarray, parameters of the Gaussian mixture
	  x:		(ndims,npoints) ndarray, evaluation points

	Outputs
	-------
	  (npoints,) ndarray, values of the Gaussian at `x`
	"""
	params = np.array(params)

	# number of dimensions
	ndims = x.shape[0]

	# number of parameters
	if covariance_parameterization=='full':
		ncov_params = ndims**2
	else:
		ncov_params = (ndims*(ndims+1))//2
	nparams = npeaks * ( 1 + ndims + ncov_params + ndims )
	if len(params)!=nparams:
		raise ValueError(f"`params` array is of wrong size, must have `len(params) = 1+npeaks*(1+ndims+ndims*(ndims+1)/2) = {nparams}`, got {len(params)}")

	# parameters of the model
	intst = params[:npeaks]**2
	cnt   = params[npeaks:(ndims+1)*npeaks]
	sqrtP = params[(ndims+1)*npeaks:npeaks*(1+ndims+ncov_params)]
	skew  = params[npeaks*(1+ndims+ncov_params):]
	skew  = np.array(skew).reshape((1,-1))

	g = 0
	for i in range(npeaks):
		# centers of gaussians
		cnt_i = cnt[i*ndims:(i+1)*ndims]
		# square root of the precision matrix
		if covariance_parameterization=='full':
			sqrtP_i = sqrtP[i*ncov_params:(i+1)*ncov_params].reshape((ndims,ndims))
		elif covariance_parameterization=='cholesky':
			sqrtP_i  = np.zeros((ndims,ndims))
			triu_ind = np.triu_indices(ndims)
			diag_ind = np.diag_indices(ndims)
			# fill upper triangular part
			sqrtP_i[triu_ind] = sqrtP[i*ncov_params:(i+1)*ncov_params]
			# positive diagonal makes cholesky decomposition unique
			sqrtP_i[diag_ind] = np.exp(sqrtP_i[diag_ind])
		elif covariance_parameterization=='givens':
			# inverse rotation matrix
			R = rotation_matrix(sqrtP[i*ncov_params:(i+1)*ncov_params-ndims])
			# square roots of the eigenvalues of the precision matrix
			sqrt_eig = sqrtP[(i+1)*ncov_params-ndims:(i+1)*ncov_params].reshape((ndims,1))
			# square root of the precision matrix
			sqrtP_i  = sqrt_eig * R
		# base Gaussian
		gi = intst[i] * np.exp(-0.5*squared_mahalanobis_distance(cnt_i, sqrtP_i, x))
		# modulated Gaussian
		gi = gi * (1+erf(skew@(x-cnt_i.reshape((ndims,1))))/np.sqrt(2)).ravel()
		# total
		g = g + gi
	return g


###############################################################################


def get_grid_data(hist_ws, bins=None, rebin_mode='density', return_edges=False):
	'''Extract coordinates of the bins and bin counts from the histogram workspace

	Inputs
	------
	  hist_ws:		histogram workspace
	  bins:			optional, list with n elements, i-th element is a list with bins in the i-th dimension
	  rebin_mode:	optional, rebinning mode: 'mass' or 'density', ignored if bins is None

	Outputs
	-------
	  points:	(n,dim_0,...,dim_n) ndarray, coordinates of the bins in n-dimensional histogram
	  data:		(dim_0,...,dim_n) ndarray, bin counts in n-dimensional histogram
	'''

	# number of dimensions in the histogram
	ndims = hist_ws.getNumDims()

	# histogram array
	data = hist_ws.getNumEventsArray().copy()

	# limits of the box along each dimension
	limits = [(hist_ws.getDimension(i).getMinimum(), hist_ws.getDimension(i).getMaximum()) for i in range(ndims)]

	# edges of the histogram along each dimension
	edges = [ np.linspace(limits[d][0],limits[d][1],data.shape[d]+1) for d in range(ndims) ]

	if bins is not None:
		# # volume of each bin
		# volume = rebin_histogram(np.ones(data.shape), bins) if rebin_mode=='density' else 1
		# counts in each bin
		data  = rebin_histogram(data, bins, mode=rebin_mode)
		# edges along each dimension in the rebinned histogram
		edges = [ edges[d][np.cumsum([0]+list(bins[d]))] for d in range(ndims) ]

	# centers of the bins along each dimension
	centers = [ 0.5*(e[:-1]+e[1:]) for e in edges ]

	# multidimensional grid of bin centers
	points = np.stack(np.meshgrid(*centers,indexing='ij'), axis=0)

	if return_edges:
		return data, points, edges
	else:
		return data, points


def initial_parameters(hist_ws, bins, detector_mask):
	'''
	Initialize parameters of Gaussian by fitting 1d marginal histograms
	'''
	# number of dimensions in the histogram
	ndims = hist_ws.getNumDims()

	# limits of the box along each dimension
	limits = [(hist_ws.getDimension(i).getMinimum(), hist_ws.getDimension(i).getMaximum()) for i in range(ndims)]

	data, points, edges = get_grid_data(hist_ws, bins, rebin_mode='density', return_edges=True)


	###########################################################################
	# estimate rotation of the ellipsoid

	# 'remove' background below 1/3 distance between max and mean intensity
	data_max, data_mean = data.max(), data.mean()
	nobkgr_data = data - (data_mean + 0.33 * (data_max - data_mean))
	nobkgr_data[nobkgr_data<0] = 0

	###################################
	# fix background
	bkgr_init = [0]
	bkgr_lbnd = [0]
	bkgr_ubnd = [0]

	# initialization and bounds for the max peak intensity
	peak_init = [np.sqrt(nobkgr_data.max())]
	peak_lbnd = [0]
	peak_ubnd = [np.inf]

	# initialization and bounds for the peak center
	dcnt     = [ (lim[1]-lim[0])/2/3 for lim in limits]
	cnt_init = [ (lim[0]+lim[1])/2   for lim in limits]
	cnt_lbnd = [c-dc for c,dc in zip(cnt_init,dcnt)]
	cnt_ubnd = [c+dc for c,dc in zip(cnt_init,dcnt)]

	# initialization and bounds for the precision matrix
	ncov_params = (ndims*(ndims+1))//2
	num_angles  = (ndims*(ndims-1))//2
	# bounds on the semiaxes of the `std` ellipsoid: standard deviations (square roots of the eigenvalues) of the covariance matrix
	peak_std = 4
	ini_rads = [ 2/3   * (lim[1]-lim[0])/2/peak_std for lim in limits]  # initial  'std' radius is 2/3   of the box radius
	max_rads = [ 5/6   * (lim[1]-lim[0])/2/peak_std for lim in limits]  # largest  'std' radius is 5/6   of the box radius
	min_rads = [ 1/64  * (lim[1]-lim[0])/2/peak_std for lim in limits]  # smallest 'std' radius is 1/64  of the box radius
	# `num_angles` angles and `ndims` singular values of the precision matrix
	# print(max(ini_rads),max_rad)
	# prec_init = [     0]*num_angles + [1/max_rad]*3
	prec_init =      [0]*num_angles + [ 1/r for r in ini_rads]
	prec_lbnd = [-np.pi]*num_angles + [ 1/r for r in max_rads]
	prec_ubnd = [ np.pi]*num_angles + [ 1/r for r in min_rads]

	# initialization and bounds for the skewness
	skew_init = [0]*ndims
	skew_lbnd = [0]*ndims
	skew_ubnd = [0]*ndims

	# all parameters
	params_init = bkgr_init + peak_init + cnt_init + prec_init + skew_init #+ tail_init + tail_cnt_init + tail_prec_init + tail_skew_init
	params_lbnd = bkgr_lbnd + peak_lbnd + cnt_lbnd + prec_lbnd + skew_lbnd #+ tail_lbnd + tail_cnt_lbnd + tail_prec_lbnd + tail_skew_lbnd
	params_ubnd = bkgr_ubnd + peak_ubnd + cnt_ubnd + prec_ubnd + skew_ubnd #+ tail_ubnd + tail_cnt_ubnd + tail_prec_ubnd + tail_skew_ubnd

	points = points.reshape((3,-1))
	points2fit = points

	def loss(params):
		fit = params[0]**2 + gaussian_mixture(params[1:],points2fit,npeaks=1,covariance_parameterization='givens').reshape(nobkgr_data.shape)
		return (fit-nobkgr_data*np.log(fit))[detector_mask].sum()
	result = minimize(loss, x0=params_init, bounds=tuple(zip(params_lbnd,params_ubnd)), method='L-BFGS-B', options={'maxiter':1000, 'maxcor':1000, 'disp':False} )
	params_fit = result.x

	# print(result.success)

	# individual parameters
	bkgr   = params_fit[0]**2
	intst  = params_fit[1]**2
	cnt    = params_fit[2:2+ndims]
	sqrtP  = params_fit[2+ndims:2+ndims+ncov_params]
	angles = sqrtP[:ndims]
	sigmas = sqrtP[ndims:]
	skew   = params_fit[2+ndims+ncov_params:]
	skew   = np.array(skew).reshape((1,-1))

	if _debug:
		from matplotlib.patches import Circle, Ellipse

		# inverse rotation matrix
		R = rotation_matrix(angles)
		# full covariance matrix
		cov_3d = R.T @ np.diag(1/sigmas**2) @ R
		# covariances and ellipsoids of 2d marginals
		cov_2d   = []
		angle_2d = []
		sigma_2d = []
		for i in range(3):
			yind, xind = [j for j in range(3) if j!=i]
			cov_2d.append( cov_3d[np.ix_([xind,yind],[xind,yind])] )
			eigi, roti = eigen(cov_2d[-1])
			angle_2d.append( np.sign(roti[0,1]) * np.arccos(roti[0,0])/np.pi*180 )
			sigma_2d.append( np.sqrt(eigi) )

		fit = params_fit[0]**2 + gaussian_mixture(params_fit[1:],points,npeaks=1,covariance_parameterization='givens')
		fit = fit.reshape(data.shape)

		# nobkgr_data = data.copy()

		data_2d = marginalize_2d(data, normalize=False)
		nobkgr_data_2d = marginalize_2d(nobkgr_data, normalize=False)
		fit_2d = marginalize_2d(fit, normalize=False)
		plt.clf()
		for i in range(ndims):
			yind, xind = [j for j in range(3) if j!=i]
			left, right, bottom, top = edges[xind][0], edges[xind][-1], edges[yind][0], edges[yind][-1]
			plt.subplot(3,ndims,i+1+0*ndims)
			plt.imshow(data_2d[i]/(data_2d[i]>0), interpolation='none', extent=(left,right,bottom,top), origin='lower')

			plt.subplot(3,ndims,i+1+1*ndims)
			# plt.gca().add_patch(Circle((cnt_init[xind],cnt_init[yind]), peak_std*max_rad,     color='red', ls='-', fill=False))
			plt.gca().add_patch(Ellipse((cnt[xind],cnt[yind]), 2*peak_std*sigma_2d[i][0], 2*peak_std*sigma_2d[i][1], angle=angle_2d[i], color='green', ls='-.', fill=False))
			plt.imshow(nobkgr_data_2d[i]/(nobkgr_data_2d[i]>0), interpolation='none', extent=(left,right,bottom,top), origin='lower')

			plt.subplot(3,ndims,i+1+2*ndims)
			# plt.gca().add_patch(Circle((cnt_init[xind],cnt_init[yind]), largest_rad, color='yellow', ls='-.', fill=False))
			# plt.gca().add_patch(Circle((cnt_init[xind],cnt_init[yind]), peak_std*max_rad,     color='red', ls='-', fill=False))
			plt.gca().add_patch(Ellipse((cnt[xind],cnt[yind]), 2*peak_std*sigma_2d[i][0], 2*peak_std*sigma_2d[i][1], angle=angle_2d[i], color='green', ls='-.', fill=False))
			plt.imshow(fit_2d[i]/(fit_2d[i]>0), interpolation='none', extent=(left,right,bottom,top), origin='lower')
		plt.savefig(f'debug/rotation.png')
		plt.clf()
	# exit()


	###########################################################################
	# background estimate

	# individual parameters
	bkgr   = params_fit[0]**2
	intst  = params_fit[1]**2
	cnt    = params_fit[2:2+ndims]
	sqrtP  = params_fit[2+ndims:2+ndims+ncov_params]
	angles = sqrtP[:ndims]
	sigmas = sqrtP[ndims:]
	skew   = params_fit[2+ndims+ncov_params:]
	skew   = np.array(skew).reshape((1,-1))

	bkgr_mask = mahalanobis_distance(cnt, sigmas.reshape((3,1))*rotation_matrix(angles), points.reshape((3,-1)))>peak_std

	params_fit[0] = np.sqrt(data.ravel()[bkgr_mask].mean())
	params_fit[1] = np.sqrt(data.ravel()[bkgr_mask].max() - params_fit[0])

	if _debug:
		from matplotlib.patches import Circle, Ellipse

		# inverse rotation matrix
		R = rotation_matrix(angles)
		# full covariance matrix
		cov_3d = R.T @ np.diag(1/sigmas**2) @ R
		# covariances and ellipsoids of 2d marginals
		cov_2d   = []
		angle_2d = []
		sigma_2d = []
		for i in range(3):
			yind, xind = [j for j in range(3) if j!=i]
			cov_2d.append( cov_3d[np.ix_([xind,yind],[xind,yind])] )
			eigi, roti = eigen(cov_2d[-1])
			angle_2d.append( np.sign(roti[0,1]) * np.arccos(roti[0,0])/np.pi*180 )
			sigma_2d.append( np.sqrt(eigi) )

		fit = params_fit[0]**2 + gaussian_mixture(params_fit[1:],points,npeaks=1,covariance_parameterization='givens')
		fit = fit.reshape(data.shape)

		# nobkgr_data = data.copy()

		data_2d = marginalize_2d(data, normalize=False)
		nobkgr_data_2d = marginalize_2d(nobkgr_data, normalize=False)
		fit_2d = marginalize_2d(fit, normalize=False)
		plt.clf()
		for i in range(ndims):
			yind, xind = [j for j in range(3) if j!=i]
			left, right, bottom, top = edges[xind][0], edges[xind][-1], edges[yind][0], edges[yind][-1]
			plt.subplot(3,ndims,i+1+0*ndims)
			plt.imshow(data_2d[i]/(data_2d[i]>0), interpolation='none', extent=(left,right,bottom,top), origin='lower')

			plt.subplot(3,ndims,i+1+1*ndims)
			plt.gca().add_patch(Ellipse((cnt[xind],cnt[yind]), 2*peak_std*sigma_2d[i][0], 2*peak_std*sigma_2d[i][1], angle=angle_2d[i], color='green', ls='-.', fill=False))
			plt.imshow(nobkgr_data_2d[i]/(nobkgr_data_2d[i]>0), interpolation='none', extent=(left,right,bottom,top), origin='lower')

			plt.subplot(3,ndims,i+1+2*ndims)
			plt.gca().add_patch(Ellipse((cnt[xind],cnt[yind]), 2*peak_std*sigma_2d[i][0], 2*peak_std*sigma_2d[i][1], angle=angle_2d[i], color='green', ls='-.', fill=False))
			plt.imshow(fit_2d[i]/(fit_2d[i]>0), interpolation='none', extent=(left,right,bottom,top), origin='lower')
		plt.savefig(f'debug/initialization.png')
		plt.clf()

	return params_fit


def fit_gaussian_multilevel(hist_ws, min_level=4, return_bins=False, loss='mle', covariance_parameterization='givens', peak_std=4, detector_mask=None, plot_intermediate=False):
	'''	'''

	# number of dimensions in the histogram
	ndims = hist_ws.getNumDims()

	# limits of the box along each dimension
	limits = [(hist_ws.getDimension(i).getMinimum(), hist_ws.getDimension(i).getMaximum()) for i in range(ndims)]

	# shape of the histogram
	shape = [hist_ws.getDimension(i).getNBins() for i in range(ndims)]

	# shape of the largest subhistogram with shape as a power of 2
	shape2 = [2**int(np.log2(shape[d])) for d in range(ndims)]

	# smallest power of 2 among all dimensions
	minpow2 = min([int(np.log2(shape[d])) for d in range(ndims)])

	# original resolution of the histogram (size of the voxel)
	resolution = [ (limits[d][1]-limits[d][0])/shape[d] for d in range(ndims)]

	params = params_lbnd = params_ubnd = None
	for p in range(min_level,minpow2+1):
		start = time.time()

		# left, middle (with power of 2 shape) and right bins
		binsl = [(shape[d]-shape2[d])//2 for d in range(ndims)]
		binsm = [split_bins([shape2[d]],2**p,recursive=False) for d in range(ndims)]
		binsr = [(shape[d]-shape2[d]) - binsl[d] for d in range(ndims)]

		# # combine two middle bins
		# binsm = [b[:len(b)//2-1]+[b[len(b)//2-1]+b[len(b)//2+1]]+b[len(b)//2+1:] for b in binsm]

		# bins at the current level
		bins = [ ([bl] if bl>0 else [])+bm+([br] if br>0 else []) for bl,bm,br in zip(binsl,binsm,binsr)]

		# fit histogram at the current level
		params, sucess, bins = fit_gaussian(hist_ws, bins=bins, return_bins=True, loss=loss, covariance_parameterization=covariance_parameterization, peak_std=peak_std, params_init=params, params_lbnd=params_lbnd, params_ubnd=params_ubnd, detector_mask=detector_mask)
		# return params, sucess, bins

		# # skip level if fit not successful
		# if not sucess:
		# 	params = params_lbnd = params_ubnd = None
		# 	continue

		nangles   = ndims*(ndims-1)//2
		sqrtbkgr  = params[0]
		sqrtintst = params[1]
		cnt       = params[2:2+ndims]
		angles    = params[2+ndims:2+ndims+nangles]
		invrads   = params[2+ndims+nangles:2+2*ndims+nangles]
		skewness  = params[2+2*ndims+nangles:2+3*ndims+nangles]

		#######################################################################
		# refine search bounds

		# bounds for the center
		dcnt = [ 3*res*2**(minpow2-p) for res in resolution] # search radius is 3 voxels at the current level
		cnt_lbnd = [c-dc for c,dc in zip(cnt,dcnt)]
		cnt_ubnd = [c+dc for c,dc in zip(cnt,dcnt)]

		# bounds for the precision matrix angles
		if minpow2>min_level:
			phi0 = np.pi/1
			phi1 = np.pi/8
			dphi = phi0 + (p-min_level)/(minpow2-min_level) * (phi1-phi0)
		else:
			dphi = np.pi

		# sc = 2 + (p-min_level)/(minpow2-min_level) * (1-2)
		# sc = 2
		# print(sc)

		# bounds for the precision matrix
		# prec_lbnd = [max(-np.pi,phi-dphi) for phi in angles] + [ r/2.0 for r in invrads]
		prec_lbnd = [max(-np.pi,phi-dphi) for phi in angles] + [ max((limits[d][1]-limits[d][0])/4/(4/3),invrads[d]/2.0) for d in range(ndims)]
		prec_ubnd = [min( np.pi,phi+dphi) for phi in angles] + [ 100*r for r in invrads]

		# bounds for all parameters
		params_lbnd = [np.abs(sqrtbkgr)/2, np.abs(sqrtintst)/2] + cnt_lbnd + prec_lbnd + [0 for sk in skewness]
		params_ubnd = [2*np.abs(sqrtbkgr), 2*np.abs(sqrtintst)] + cnt_ubnd + prec_ubnd + [0 for sk in skewness]

		params[0] = np.abs(sqrtbkgr)
		params[1] = np.abs(sqrtintst)

		# params_lbnd = params_lbnd + list(params[len(params_lbnd):])
		# params_ubnd = params_ubnd + list(params[len(params_ubnd):])
		# params_lbnd = params_ubnd = None

		#######################################################################

		print(f"Fitting histogram with {2**p:3d} bins: {time.time()-start:.3f} seconds")

		#if plot_intermediate:
		#	plot_fit(hist_ws, params, bins, prefix=f"{p}", peak_id=1074, peak_hkl=[2.0,-2.0,-9.0], peak_std=4, bkgr_std=7, detector_mask=None, log=True)

	return fit_gaussian(hist_ws, return_bins=return_bins, loss=loss, covariance_parameterization=covariance_parameterization, peak_std=peak_std, params_init=params, params_lbnd=params_lbnd, params_ubnd=params_ubnd, detector_mask=detector_mask)


def fit_gaussian(hist_ws, bins=None, return_bins=False, loss='mle', covariance_parameterization='givens', peak_std=4, detector_mask=None, cnt_init=None, dcnt=None, params_init=None, params_lbnd=None, params_ubnd=None):
	'''
	Inputs
	------
	  hist_ws:	histogram workspace
	  bins:		rebinning algorithm, one of [None, 'knuth', 'adaptive_knuth', int, list]
	  loss:		fitting criterion, one of ['pearson_chi', 'neumann_chi', 'mle']
	  cnt:		initial estimate for the peak center
	  dcnt:		bounds for the location of the peak center

	Outputs
	-------
	  (1+(1+ndims+ndims**2)*npeaks,) ndarray of parameters
	'''

	# number of dimensions in the histogram
	ndims      = hist_ws.getNumDims()
	hist_shape = hist_ws.getNumEventsArray().shape

	# limits of the box along each dimension
	limits = [(hist_ws.getDimension(i).getMinimum(), hist_ws.getDimension(i).getMaximum()) for i in range(ndims)]

	# rebin points and data
	if isinstance(bins,str):
		# get points to fit
		data, points, edges = get_grid_data(hist_ws, return_edges=True)

		if bins=='knuth':
			bins = knuth_bins(data, min_bins=4, spread=1)
			# bins = knuth_bins(data, min_bins=4, max_bins=4, spread=0)
		elif bins=='adaptive_knuth':
			# rebin data using number of bins given by `knuth` algorithm but such that bins have comparable probability masses
			bins  = knuth_bins(data, min_bins=4, spread=1)

			# 1d marginals
			marginal_data = marginalize_1d(data, normalize=False)

			# quantiles, note len(b)+2 to make odd number of bins
			quant = [ np.linspace(0,1,min(len(b)+2,data.shape[i])) for i,b in enumerate(bins) ]
			# edges = [ np.percentile( np.repeat(np.arange(1,md.size+1), md.astype(int)), 100*q[1:], interpolation='higher' ) for md,q in zip(marginal_data,quant) ]
			# bins  = [ np.diff([0]+list(e)).astype(int) for e in edges ]
			edges = [ np.quantile( np.repeat(np.arange(1,md.size+1), md.astype(int)), q[1:], method='inverted_cdf' ) for md,q in zip(marginal_data,quant) ]
			bins  = [ np.diff(e,prepend=0).astype(int) for e in edges ]

	elif isinstance(bins,int):
		nbins = bins
		bins  = [split_bins([s],nbins,recursive=False) for s in hist_shape]
	elif bins is None:
		bins = [[1]*s for s in hist_shape]

	# rebinned data
	data, points = get_grid_data(hist_ws, bins, rebin_mode='density')
	# vol          = bins_volume(bins)
	# count_data   = vol * data

	if loss in ['pearson_chi','neumann_chi']:
		# no zero bins allowed
		nnz_mask = data>0
		# nonzero data bins
		nnz_data = data[nnz_mask]

	# reshape points to (ndims,npoints)
	points = points.reshape((ndims,-1))

	# detector_mask = None
	if detector_mask is None:
		detector_mask = (data==data)
	else:
		detector_mask = (rebin_histogram(detector_mask.astype(int), bins)>0).astype(int)
		# detector_mask = 1 - detector_mask
		detector_mask = (detector_mask!=0) #.astype(int)
		# detector_mask = np.ones_like(detector_mask)
		# detector_mask[:10,:10,:10] = 0

	###########################################################################
	# initialization and bounds on parameters

	if (params_init is None) or (params_lbnd is None) or (params_ubnd is None):
		data_min, data_max, data_mean = data.min(), data.max(), data.mean()
		# intst_max = data_max-data_min + 1	# add 1 to ensure intst_max>0

		###################################
		# initialization and bounds for the background intensity
		bkgr_init = [np.sqrt(0.9*data_mean)]
		bkgr_lbnd = [0]
		bkgr_ubnd = [np.sqrt(data_mean)]

		###################################
		# initialization and bounds for the max peak intensity
		peak_init = [np.sqrt(data_max-data_min)]
		peak_lbnd = [0]
		peak_ubnd = [np.sqrt(2*data_max)]


		###################################
		# cnt_1d, std_1d = initial_parameters(hist_ws, bins)
		params_init1 = initial_parameters(hist_ws, bins, detector_mask)

		# initialization and bounds for the peak center
		if cnt_init is None:
			cnt_init = [ (lim[0]+lim[1])/2  for lim in limits]
			# cnt_init = cnt_1d
		if dcnt is None:
			dcnt = [ (lim[1]-lim[0])/6  for lim in limits]
		cnt_lbnd = [c-dc for c,dc in zip(cnt_init,dcnt)]
		cnt_ubnd = [c+dc for c,dc in zip(cnt_init,dcnt)]

		# initialization and bounds for the precision matrix
		if covariance_parameterization=='givens':
			num_angles = (ndims*(ndims-1))//2
			# bounds on the semiaxes of the `std` ellipsoid: standard deviations (square roots of the eigenvalues) of the covariance matrix
			#std = 4.0
			ini_rads = [ 1/4   * (lim[1]-lim[0])/2/peak_std for lim in limits]  # initial  'peak_std' radius is 1/4   of the box radius
			max_rads = [ 3/4   * (lim[1]-lim[0])/2/peak_std for lim in limits]  # largest  'peak_std' radius is 3/4   of the box radius
			min_rads = [ 1/32 * (lim[1]-lim[0])/2/peak_std for lim in limits]  # smallest 'peak_std' radius is 1/32 of the box radius
			# `num_angles` angles and `ndims` square roots of the eigenvalues of the precision matrix
			prec_init = [     0]*num_angles + [ 1/r for r in ini_rads]
			# prec_init = [     0]*num_angles + std_1d
			prec_lbnd = [-np.pi]*num_angles + [ 1/r for r in max_rads]
			prec_ubnd = [ np.pi]*num_angles + [ 1/r for r in min_rads]
		elif covariance_parameterization=='cholesky':
			num_chol = (ndims*(ndims+1))//2
			# upper triangular part of the Cholesky factor of the precision matrix
			prec_init = list(np.eye(ndims)[np.triu_indices(ndims)])
			prec_lbnd = [-1000]*num_chol
			prec_ubnd = [ 1000]*num_chol
		elif covariance_parameterization=='full':
			# arbitrary square root of the precision matrix
			prec_init = list(np.eye(ndims).ravel())
			prec_lbnd = [-1000]*(ndims**2)
			prec_ubnd = [ 1000]*(ndims**2)

		# initialization and bounds for the skewness
		skew_init = [0]*ndims
		skew_lbnd = [0]*ndims
		skew_ubnd = [0]*ndims


	###################################
	# initialization and bounds for all parameters
	if params_init is None:
		params_init = bkgr_init + peak_init + cnt_init + prec_init + skew_init #+ tail_init + tail_cnt_init + tail_prec_init + tail_skew_init
		params_init = params_init1
	if params_lbnd is None: params_lbnd = bkgr_lbnd + peak_lbnd + cnt_lbnd + prec_lbnd + skew_lbnd #+ tail_lbnd + tail_cnt_lbnd + tail_prec_lbnd + tail_skew_lbnd
	if params_ubnd is None: params_ubnd = bkgr_ubnd + peak_ubnd + cnt_ubnd + prec_ubnd + skew_ubnd #+ tail_ubnd + tail_cnt_ubnd + tail_prec_ubnd + tail_skew_ubnd


	if loss=='pearson_chi':
		def residual(params):
			fit = params[0]**2
			fit = fit + gaussian_mixture(params[1:],points,npeaks=1,covariance_parameterization=covariance_parameterization).reshape(data.shape)
			res = fit[nnz_mask] - nnz_data
			return (res/np.sqrt(fit)).ravel()
		result = least_squares(residual, #jac=jacobian_residual,
			x0=params_init, bounds=[params_lbnd,params_ubnd], method='trf', verbose=0, max_nfev=1000)
	elif loss=='neumann_chi':
		def residual(params):
			fit = params[0]**2
			fit = fit + gaussian_mixture(params[1:],points,npeaks=1,covariance_parameterization=covariance_parameterization).reshape(data.shape)
			res = fit[nnz_mask] - nnz_data
			return (res/np.sqrt(nnz_data)).ravel()
			# return (res/np.maximum(1,np.sqrt(nnz_data))).ravel()
			# return (res/np.sqrt(data[nnz_mask].size)).ravel()
		result = least_squares(residual, #jac=jacobian_residual,
			x0=params_init, bounds=[params_lbnd,params_ubnd], method='trf', verbose=0, max_nfev=1000)
	elif loss=='mle':
		# def tail_loss(params):
		# 	fit = gaussian_mixture(params, points,npeaks=1,covariance_parameterization=covariance_parameterization).reshape(data.shape)
		# 	return (fit-resid*np.log(fit))[detector_mask].sum()
		def peak_loss(params):
			fit = params[0]**2
			fit = fit + gaussian_mixture(params[1:],points,npeaks=1,covariance_parameterization=covariance_parameterization).reshape(data.shape)
			return (fit-data*np.log(fit))[detector_mask].sum()
		result = minimize(peak_loss,
			x0=params_init, bounds=tuple(zip(params_lbnd,params_ubnd)),
			method='L-BFGS-B', options={'maxiter':1000, 'maxcor':1000, 'disp':False}
			)

		fit_params = result.x

		print(result.success)


	if return_bins:
		return fit_params, result.success, bins
	else:
		return fit_params, result.success

###############################################################################


def plot_fit(hist_ws, gauss_params, bins, plot_path='output', prefix=None, peak_id=None, peak_hkl=None, peak_std=4, bkgr_std=9, detector_mask=None, log=False):
	from matplotlib.patches import Ellipse

	# create output directory for plots
	from pathlib import Path
	Path(plot_path).mkdir(parents=True, exist_ok=True)

	# fit model to data
	data, points, edges = get_grid_data(hist_ws, return_edges=True)
	# gauss_params, bins  = fit_gaussian(hist_ws, bins='adaptive_knuth', loss='pearson_chi', return_bins=True, covariance_parameterization='givens')

	# point along each dimension
	dim_points = [points[0][:,0,0],points[1][0,:,0],points[2][0,0,:]]

	# parameters of the model
	bkgr     = gauss_params[0]**2
	intst    = gauss_params[1]**2
	mu       = gauss_params[2:5]
	angles   = gauss_params[5:8]
	sqrt_eig = 1 / np.array(gauss_params[8:11])
	# sqrtP = gauss_params[5:].reshape((3,3))
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
		roti,eigi,_ = svd(cov_2d[-1])
		# eigi, roti = eig(cov_2d[-1])
		# eigi, roti = eigen(cov_2d[-1])
		angle_2d.append( np.sign(roti[0,1]) * np.arccos(roti[0,0])/np.pi*180 )
		sigma_2d.append( np.sqrt(eigi) )

	# fitted model
	fit = bkgr + gaussian_mixture(gauss_params[1:], points.reshape((3,-1)), covariance_parameterization='givens').reshape(data.shape)
	fit_masked = (1 if detector_mask is None else detector_mask) * fit

	# rebinned data and fit
	rebinned_data, rebinned_points, rebinned_edges = get_grid_data(hist_ws, bins, rebin_mode='density', return_edges=True)
	rebinned_fit = rebin_histogram(fit, bins, mode='density')


	########################################
	# plot 1d marginals
	normalize = False
	data_1d = marginalize_1d(data, normalize=normalize, detector_mask=detector_mask)
	fit_1d  = marginalize_1d(fit,  normalize=normalize, detector_mask=detector_mask)
	rebinned_data_1d = marginalize_1d(rebinned_data, normalize=normalize, bins=bins, detector_mask=detector_mask)
	rebinned_fit_1d  = marginalize_1d(rebinned_fit,  normalize=normalize, bins=bins, detector_mask=detector_mask)
	# exit()

	fig = plt.figure(constrained_layout=True, figsize=(20,45))
	axes = fig.subplots(7,3)
	# fig = plt.figure(constrained_layout=True, figsize=(20,35))
	# axes = fig.subplots(5,3)
	ax_id = -1
	##############################
	ax_id += 1
	for i,ax in enumerate(axes[ax_id]):
		ax.stairs(data_1d[i], edges=edges[i], fill=True)
		ax.stairs(rebinned_data_1d[i], edges=rebinned_edges[i], fill=False, lw=1.5)
		ax.plot(dim_points[i], fit_1d[i])
		ax.vlines([mu[i]-peak_std*np.sqrt(cov_3d[i,i]),mu[i]+peak_std*np.sqrt(cov_3d[i,i])], 0, data_1d[i].max(), color='r', ls='-')
		ax.vlines([mu[i]-bkgr_std*np.sqrt(cov_3d[i,i]),mu[i]+bkgr_std*np.sqrt(cov_3d[i,i])], 0, data_1d[i].max(), color='r', ls='-.')
		ax.set_xlabel(hist_ws.getDimension(i).name, fontsize='x-large')
		if i==0:
			ax.legend(['data','reb. data','fit', f'{peak_std} sigma', f'{bkgr_std} sigma'], framealpha=1.0, fontsize='xx-large')
		ax.set_box_aspect(1)

	ax_id += 1
	for i,ax in enumerate(axes[ax_id]):
		ax.stairs(rebinned_data_1d[i], edges=rebinned_edges[i], fill=True)
		ax.stairs(rebinned_fit_1d[i],  edges=rebinned_edges[i], fill=False, lw=1.5)
		ax.plot(dim_points[i], fit_1d[i])
		ax.set_xlabel(hist_ws.getDimension(i).name, fontsize='x-large')
		if i==0:
			ax.legend(['reb. data','reb. fit', 'fit'], fontsize='xx-large')
		ax.set_box_aspect(1)

	# plt.savefig(f'{plot_path}/peak_[{peak_hkl[0]},{peak_hkl[1]},{peak_hkl[2]}]_number_{peak_id}_1d.png')

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
	# # subfig2.suptitle('Gaussian fit with peak/background regions for (sigma/I) criterion')
	# plt.savefig(f'{plot_path}/peak_[{peak_hkl[0]},{peak_hkl[1]},{peak_hkl[2]}]_number_{peak_id}_2d.png')


	# fig = plt.figure(constrained_layout=True, figsize=(10,6))
	# axes = fig.subplots(2,3)
	# rebinned data
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
	# plt.savefig(f'{plot_path}/peak_[{peak_hkl[0]},{peak_hkl[1]},{peak_hkl[2]}]_number_{peak_id}_2d_rebin.png')



	########################################
	# plot difference

	ax_id += 1
	for i,ax in enumerate(axes[ax_id]):
		yind, xind = [j for j in range(3) if j!=i]
		left, right, bottom, top = edges[xind][0], edges[xind][-1], edges[yind][0], edges[yind][-1]
		ax.imshow(np.abs(fit_masked_2d[i]-data_2d[i]), interpolation='none', extent=(left,right,bottom,top), origin='lower')
		ax.set_xlabel(hist_ws.getDimension(xind).name, fontsize='x-large')
		ax.set_ylabel(hist_ws.getDimension(yind).name, fontsize='x-large')

	# save
	if prefix is None:
		plt.savefig(f'{plot_path}/{peak_id}.png')
	else:
		plt.savefig(f'{plot_path}/{prefix}_number_{peak_id}.png')

	plt.close('all')


def integrate_peak(hist_ws, params, detector_mask=None, peak_estimate='data', background_estimate='data', peak_std=4, bkgr_std=None):
	r'''Integrate peak intensity with background correction

	Inputs
	------
	  hist_ws:	histogram workspace of the peak
	  params:	parameters of the fit
	  peak_estimate:		how to calculate peak,       one of ['fit','data']
	  background_estimate:	how to calculate background, one of ['fit','data']

	Output
	------
	  corrected_intensity
	  corrected_sigma
	'''

	if bkgr_std is None: bkgr_std = peak_std + 4

	# parameters of the model
	bkgr   = params[0]**2
	intst  = params[1]**2
	mu     = params[2:5]
	angles = params[5:8]
	svals  = np.array(params[8:11]).reshape((3,1))

	# inverse rotation matrix
	R = rotation_matrix(angles)

	#
	data, points, edges = get_grid_data(hist_ws, return_edges=True)
	points = points.reshape((3,-1))
	fit = params[0]**2 + gaussian_mixture(params[1:],points,npeaks=1,covariance_parameterization='givens').reshape(data.shape)

	data = data.ravel()
	fit  = fit.ravel()

	# detector_mask = None
	if detector_mask is None:
		detector_mask = (data==data)
	else:
		detector_mask = detector_mask.astype(bool)

	mah_dist = mahalanobis_distance(mu, svals*R, points)

	# true distance mask
	# dist_mask = np.sqrt(np.sum((points.reshape((3,-1))-mu.reshape((3,1)))**2,axis=0)) < 0.3

	# mahalanobis distance masks
	peak_mask = np.logical_and(detector_mask.ravel(), mah_dist<peak_std)
	bkgr_mask = np.logical_and(detector_mask.ravel(), mah_dist>peak_std)
	bkgr_mask = np.logical_and(bkgr_mask,mah_dist<bkgr_std)
	# bkgr_mask = np.logical_and(bkgr_mask,dist_mask)

	peak_vol  = peak_mask.sum()
	bkgr_vol  = bkgr_mask.sum()
	peak2bkgr = peak_vol / bkgr_vol

	# total peak intensity
	if peak_estimate=='data':
		total_peak_intensity = data[peak_mask].sum()
	elif peak_estimate=='fit':
		total_peak_intensity = fit[peak_mask].sum()

	# background correction
	if background_estimate=='data':
		total_bkgr_intensity = data[bkgr_mask].sum()
	elif background_estimate=='fit':
		total_bkgr_intensity = bkgr
	peak_bkgr_correction = peak2bkgr    * total_bkgr_intensity
	peak_bkgr_variance   = peak2bkgr**2 * total_bkgr_intensity

	peak_chi2 = ((data-fit)**2/fit)[peak_mask].mean()
	# print('Chi2: ',chi2)

	intensity = total_peak_intensity - peak_bkgr_correction
	sigma     = total_peak_intensity + peak_bkgr_variance
	if peak_estimate=='fit':
		sigma = sigma + ((data-fit)**2/(data+1))[peak_mask].sum()
	sigma = np.sqrt(sigma)

	return intensity, sigma, peak_chi2, total_bkgr_intensity



if __name__ == '__main__':
	run = 43652
	peaks_ws = LoadIsawPeaks(f'{run}_Niggli.integrate')

	loss = 'mle'
	bins = 32
	n_std = 4

	# for peak_id in [2,18,84,200,1074,1077,1082]:
	for peak_id in [1077,1082]:
	# for peak_id in [38,63,74]:
	# for peak_id in range(0,100):

		print(f'Processing peak {peak_id}, loss {loss}')
		peak = peaks_ws.getPeak(peak_id)
		hkl  = peak.getHKL()

		hist_ws = LoadMD(f'TOPAZ_{run}_peak_{peak_id}.nxs')
		# data = hist_ws.getNumEventsArray().copy()
		# np.save(f'peak_{peak_id}.npy',data)
		# continue

		detector_mask = np.ones_like(hist_ws.getSignalArray())
		detector_mask[:64,:,:] = 0
		plt.imsave('tmp0.png', detector_mask.sum(axis=0))
		plt.imsave('tmp1.png', detector_mask.sum(axis=1))
		plt.imsave('tmp2.png', detector_mask.sum(axis=2))

		# print( peaks_ws.createPeakHKL([0,0,0]) )
		# exit()

		# fit model
		# gauss_params, fit_bins = fit_gaussian(hist_ws, bins=bins, loss=loss, return_bins=True, covariance_parameterization='givens')
		gauss_params, fit_bins = fit_gaussian_multilevel(hist_ws, bins_min=4, bins_max=32, return_bins=True, loss=loss, covariance_parameterization='givens', peak_std=n_std, detector_mask=detector_mask)

		# estimated center of the peak
		center = gauss_params[2:5]

		# integrate peak
		intensity, sigma = integrate_peak(hist_ws, detector_mask, gauss_params, n_std=n_std)

		# plot the fit
		plot_fit(hist_ws, gauss_params, fit_bins, peak_id=peak_id, peak_hkl=hkl, n_std=n_std)

		peak.setIntensity(intensity)
		peak.setSigmaIntensity(sigma)
		peak.setQSampleFrame(V3D(center[0],center[1],center[2]))

	SaveIsawPeaks( InputWorkspace=peaks_ws, AppendFile=False, Filename='./{0:d}_Niggli_4sig.integrate'.format(run), RenumberPeaks=True )
