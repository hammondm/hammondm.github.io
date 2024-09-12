
from scipy.io import wavfile
import numpy as np
import librosa

def lpc(y,order):
	axis=-1
	#reorient data
	y = y.swapaxes(axis,0)
	#get datatype
	dtype = y.dtype
	shape = list(y.shape)
	shape[0] = order + 1
	#set up autoregression coefficients
	ar_coeffs = np.zeros(
		tuple(shape),
		dtype=dtype
	)
	ar_coeffs[0] = 1
	ar_coeffs_prev = ar_coeffs.copy()
	shape[0] = 1
	#set up reflection coefficients
	reflect_coeff = np.zeros(
		shape,
		dtype=dtype
	)
	#denominator
	den = reflect_coeff.copy()
	epsilon = np.finfo(den.dtype).tiny
	#set up error in both directions
	fwd_pred_error = y[1:]
	bwd_pred_error = y[:-1]
	den[0] = np.sum(
		fwd_pred_error**2 + bwd_pred_error**2,
		axis=0
	)
	#fill out reflection coefficients
	for i in range(order):
		reflect_coeff[0] = np.sum(
			bwd_pred_error * fwd_pred_error,
			axis=0
		)
		reflect_coeff[0] *= -2
		reflect_coeff[0] /= den[0] + epsilon
		ar_coeffs_prev,ar_coeffs = \
			ar_coeffs,ar_coeffs_prev
		#go back to forward coefficients
		for j in range(1,i + 2):
			ar_coeffs[j] = (
				ar_coeffs_prev[j] + \
					reflect_coeff[0] * \
					ar_coeffs_prev[i - j + 1]
			)
		#update error
		fwd_pred_error_tmp = fwd_pred_error
		fwd_pred_error = fwd_pred_error + \
			reflect_coeff * bwd_pred_error
		bwd_pred_error = bwd_pred_error + \
			reflect_coeff * fwd_pred_error_tmp
		q = 1.0 - reflect_coeff[0] ** 2
		#update denominator
		den[0] = q * den[0] - bwd_pred_error[-1] \
			** 2 - fwd_pred_error[0] ** 2
		fwd_pred_error = fwd_pred_error[1:]
		bwd_pred_error = bwd_pred_error[:-1]
	return np.swapaxes(ar_coeffs,0,axis)

