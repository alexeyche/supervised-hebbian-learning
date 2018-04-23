
import numpy as np

def cmds_cost(x, y):
	x_gram = np.dot(x, x.T)
	y_gram = np.dot(y, y.T)
	return np.sum(np.square(y_gram - x_gram))

def eigvec_cost(W, M, eigvec):
	F = np.dot(W, np.eye(M.shape[0]) + np.linalg.inv(M))

	_, eigvecF = np.linalg.eig(np.dot(F, F.T))
	return np.sum(np.square(np.abs(eigvecF) - np.abs(eigvec)))

def eigvec_cost_gram(eigvecGx, y):
	Gy = np.dot(y, y.T) / y.shape[0] / y.shape[1]
	eigvalGy, eigvecGy = np.linalg.eig(Gy)
	eigvecGy = np.real(eigvecGy[:, :eigvecGx.shape[1]])
	# assert np.real(np.abs(eigvalGy[eigvecGx.shape[1]])) < 1e-10, \
	# 	"Bad results of eigenvalue decomposition of Y: {}".format(
	# 		np.real(np.abs(eigvalGy[eigvecGx.shape[1]]))
	# 	)

	return np.sum(np.square(np.abs(eigvecGy) - np.abs(eigvecGx)))

def phi_capital(C, p, k):
	return np.sum(np.square(np.sum(C, axis=0) - p)) * k/2.0

def lateral_cost(L, Ldiag, p, q):
	return np.sum(np.square(L - p)) + np.sum(np.square(Ldiag - q))

def correlation_cost(W, syn, y):
	return np.sum(W * np.dot(syn.T, y))


def orthonormal_cost(W, M):
	F = np.dot(W, np.eye(M.shape[0]) + np.linalg.inv(M))

	return np.linalg.norm(np.dot(F.T, F) - np.eye(M.shape[0]))
	
