

import numpy as np
from poc.opt import *
from datasets import *
from poc.common import *
from cost import *
from sklearn import datasets as sklearn_datasets
from sklearn.feature_extraction.image import extract_patches_2d

np.random.seed(12)
rng = np.random.RandomState(12)

faces = sklearn_datasets.fetch_olivetti_faces()
image_num = faces.images.shape[0]
image_num = 50  # TODO

patch_size = (20, 20)
max_patches = 50

data = np.zeros((max_patches * image_num, patch_size[0]*patch_size[1]))
target = np.zeros((max_patches * image_num,))
for img_id, img in enumerate(faces.images):
    if img_id >= image_num:
        break

    patches_id = ((img_id * max_patches),((img_id+1) * max_patches))
    
    data[patches_id[0]:patches_id[1], :] = extract_patches_2d(
        img, 
        patch_size, 
        max_patches=max_patches, 
        random_state=rng
    ).reshape((max_patches, patch_size[0]*patch_size[1]))
    
    target[patches_id[0]:patches_id[1]] = faces.target[img_id]

output = one_hot_encode(target)



data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
data *= 0.2
data = np.maximum(data - np.mean(data, 0), 0.0)


data_size, input_size = data.shape
data_size, output_size = output.shape

act = Relu()

batch_size = 250
seq_length = 50
layer_size = 50

seed = 10
layer_size = 50
dt = 0.25
k = 1.0
num_iters = 50

p, q = 0.001, 0.09
omega = p/0.1



W = positive_random_norm(input_size, layer_size, p)

Wo = positive_random_norm(layer_size, output_size, p)

L = np.zeros((layer_size, layer_size))
Ldiag = np.ones((layer_size,))

Lo = np.zeros((output_size, output_size))
Lodiag = np.ones((output_size,))


D = np.ones((layer_size, layer_size)) * p
np.fill_diagonal(D, q)

opt = SGDOpt((
	0.0002, 0.01, 0.01, 
	0.0002, 0.01, 0.01
))
opt.init(W, L, Ldiag) #, Wo, Lo, Lodiag)
Wcp = W.copy()

epochs = 1000
metrics = np.zeros((epochs, 7))
Ch = np.zeros((epochs, input_size, layer_size))
nidx = np.arange(layer_size)
for e in xrange(epochs):

	metrics_it = np.zeros((num_iters, 5))

	yh = np.zeros((num_iters, batch_size, layer_size))
	yoh = np.zeros((num_iters, batch_size, output_size))
	lat_h = np.zeros((num_iters, batch_size, layer_size))
	fb_h = np.zeros((num_iters, batch_size, layer_size))
	ff_h = np.zeros((num_iters, batch_size, layer_size))
	
	dW = np.zeros(W.shape)
	dL = np.zeros(L.shape)
	dLdiag = np.zeros(Ldiag.shape)

	dWo = np.zeros(Wo.shape)
	dLo = np.zeros(Lo.shape)
	dLodiag = np.zeros(Lodiag.shape)

	
	for bi in xrange(data.shape[0]/batch_size):
		xv = data[(bi*batch_size):((bi+1)*batch_size)]

		y = np.zeros((batch_size, layer_size))
		yo = np.zeros((batch_size, output_size))
		syn = np.zeros((batch_size, input_size))
		osyn = np.zeros((batch_size, output_size))

		ff = np.dot(xv, W) 
		# for ni in xrange(layer_size):
		# 	ni_other = np.where(nidx != ni)[0]
		# 	y[:,ni] += dt * (
		# 		act((ff[:,ni] - np.sum(np.dot(y[:,ni_other], L[ni_other, ni_other]))) / Ldiag[ni]) )

		y = act((ff - np.dot(y, L)) / Ldiag)

		# for t in xrange(num_iters):	
		# 	y += dt * (act((ff - np.dot(y, L)) / Ldiag) ) #- y)
			
		# 	ff_yo = np.dot(y, Wo)
		# 	# ff_yo = ff_yo/np.linalg.norm(ff_yo)
			
		# 	# fb_yo = yv
		# 	# fb_yo = fb_yo/np.linalg.norm(fb_yo)
			
		# 	yo += dt * (act((ff_yo - np.dot(yo, Lo))/ Lodiag) - yo)

		# 	yh[t] = y.copy()
		# 	yoh[t] = yo.copy()


		dW += np.dot(xv.T, y) - k * (np.sum(W, axis=0) - p)
		dL += np.dot(y.T, y) #- p * p
		dLdiag += np.sum(np.square(y), 0) - q * q

		# dWo += np.dot(y.T, yv) - Wo #k * (np.sum(Wo, axis=0) - p)
		# dLo += np.dot(yo.T, yo) - np.eye(output_size)
		# dLodiag += np.sum(np.square(yv), 0) - q * q

	opt.update(-dW, -dL, -dLdiag) #, -dWo, -dLo, -dLodiag)

	np.fill_diagonal(L, 0.0)
	W = np.minimum(np.maximum(W, 0.0), omega)
	L = np.maximum(L, 0.0)
	Ldiag = np.maximum(Ldiag, 0.0)


	np.fill_diagonal(Lo, 0.0)
	Wo = np.minimum(np.maximum(Wo, 0.0), omega)
	Lo = np.maximum(Lo, 0.0)
	Lodiag = np.maximum(Lodiag, 0.0)

	opt.init(W, L, Ldiag) #, Wo, Lo, Lodiag)

	metrics[e, :4] = (
		correlation_cost(W, xv, y),
		phi_capital(W, p, k),
		lateral_cost(L, Ldiag, p, q),
		np.linalg.norm(ff - y)
	)
	Ch[e] = np.dot(xv.T, y)
	
	if e % 10 == 0:

		print "Epoch {}, {}".format(
			e,
			", ".join(["{:.4f}".format(m) for m in metrics[e, :]])
		)



shm(
	W,
	W[:,0].reshape((20, 20)),
	W[:,10].reshape((20, 20)),
	W[:,25].reshape((20, 20)),
	W[:,-5].reshape((20, 20)),
	W[:,-1].reshape((20, 20))
)