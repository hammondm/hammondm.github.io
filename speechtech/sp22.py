import scipy.linalg as la
import numpy as np

def lpc2formants(lpc,sr):
	#solve as polynomial
	roots = np.roots(lpc)
	pos_roots = roots[np.imag(roots)>=0]
	#make sure there are enough
	if len(pos_roots)<len(roots)//2:
		pos_roots = list(pos_roots) + [0] * \
			(len(roots)//2 - len(pos_roots))
	#make sure there aren't too many
	if len(pos_roots)>len(roots)//2:
		pos_roots = pos_roots[:len(roots)//2]
	w = np.angle(pos_roots)
	a = np.abs(pos_roots)
	order = np.argsort(w)
	w = w[order]
	a = a[order]
	freqs = w * (sr/(2*np.pi))
	bws = -0.5 * (sr/(2*np.pi)) * np.log(a)
	return freqs,bws
