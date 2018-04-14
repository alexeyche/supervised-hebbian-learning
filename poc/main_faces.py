

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
max_patches = 20

xv = np.zeros((max_patches * image_num, patch_size[0]*patch_size[1]))
yv = np.zeros((max_patches * image_num,))
for img_id, img in enumerate(faces.images):
    if img_id >= image_num:
        break

    patches_id = ((img_id * max_patches),((img_id+1) * max_patches))
    
    xv[patches_id[0]:patches_id[1], :] = extract_patches_2d(
        img, 
        patch_size, 
        max_patches=max_patches, 
        random_state=rng
    ).reshape((max_patches, patch_size[0]*patch_size[1]))
    
    yv[patches_id[0]:patches_id[1]] = faces.target[img_id]

yv = one_hot_encode(yv)

# xv = np.maximum(xv - np.mean(xv, 0), 0.0)

# xv = np.asarray([[1.0, 0.0], [0.0, 1.0]])
# yv = np.asarray([[1.0, 0.0], [0.0, 1.0]])

xv -= np.mean(xv, axis=0)
xv /= np.std(xv, axis=0)
xv *= 0.1


data_size, input_size = xv.shape
data_size, output_size = yv.shape


(batch_size, input_size), (_, output_size) = xv.shape, yv.shape

xtv = xv[:batch_size].copy()
ytv = yv[:batch_size].copy()

act = Relu()


seq_length = 50
layer_size = 50

seed = 10
layer_size = 50
dt = 0.25
k = 1.0
num_iters = 50

p, q = 0.05, 0.01
omega = p/0.1



np.random.seed(seed)


# yv *= 0.05

W = positive_random_norm(input_size, layer_size, p)

Wo = positive_random_norm(layer_size, output_size, p)

L = np.zeros((layer_size, layer_size))
Ldiag = np.ones((layer_size,))

Lo = np.zeros((output_size, output_size))
Lodiag = np.ones((output_size,))


D = np.ones((layer_size, layer_size)) * p
np.fill_diagonal(D, q)

opt = SGDOpt((
	0.001, 0.01, 0.01, 
	0.001, 0.01, 0.01
))
opt.init(W, L, Ldiag, Wo, Lo, Lodiag)
Wcp = W.copy()

epochs = 10
metrics = np.zeros((epochs, 7))
Ch = np.zeros((epochs, input_size, layer_size))
for e in xrange(epochs):
	yh = np.zeros((num_iters, batch_size, layer_size))
	yoh = np.zeros((num_iters, batch_size, output_size))
	lat_h = np.zeros((num_iters, batch_size, layer_size))
	fb_h = np.zeros((num_iters, batch_size, layer_size))
	ff_h = np.zeros((num_iters, batch_size, layer_size))

	y = np.zeros((batch_size, layer_size))
	yo = np.zeros((batch_size, output_size))
	syn = np.zeros((batch_size, input_size))
	osyn = np.zeros((batch_size, output_size))

	metrics_it = np.zeros((num_iters, 5))

	fb_ap = np.zeros((batch_size, layer_size)) #
	
	ff = np.dot(xv, W) 
	# ff = ff/np.linalg.norm(ff)

	# fb_ap = np.dot(yv, Wo.T)
	# fb_ap = fb_ap/np.linalg.norm(fb_ap)

	for t in xrange(num_iters):	
		y += dt * (act((ff - np.dot(y, L)) / Ldiag) - y)
		
		ff_yo = np.dot(y, Wo)
		# ff_yo = ff_yo/np.linalg.norm(ff_yo)
		
		# fb_yo = yv
		# fb_yo = fb_yo/np.linalg.norm(fb_yo)
		
		yo += dt * (act((ff_yo - np.dot(yo, Lo))/ Lodiag) - yo)

		yh[t] = y.copy()
		yoh[t] = yo.copy()


	dW = np.dot(xv.T, y) - k * (np.sum(W, axis=0) - p)
	dL = np.dot(y.T, y) - p * p
	dLdiag = np.sum(np.square(y), 0) - q * q

	dWo = np.dot(y.T, yv) - Wo #k * (np.sum(Wo, axis=0) - p)
	dLo = np.dot(yo.T, yo) - np.eye(output_size)
	dLodiag = np.sum(np.square(yv), 0) - q * q

	opt.update(-dW, -dL, -dLdiag, -dWo, -dLo, -dLodiag)

	np.fill_diagonal(L, 0.0)
	W = np.minimum(np.maximum(W, 0.0), omega)
	L = np.maximum(L, 0.0)
	Ldiag = np.maximum(Ldiag, 0.0)


	np.fill_diagonal(Lo, 0.0)
	Wo = np.minimum(np.maximum(Wo, 0.0), omega)
	Lo = np.maximum(Lo, 0.0)
	Lodiag = np.maximum(Lodiag, 0.0)

	opt.init(W, L, Ldiag, Wo, Lo, Lodiag)

	yt = np.zeros((xtv.shape[0], layer_size))
	yto = np.zeros((xtv.shape[0], output_size))

	fft = np.dot(xtv, W)
	
	for t in xrange(num_iters):
		yt += dt * (act((fft - np.dot(yt, L)) / Ldiag) - yt)
		yto += dt * (act((np.dot(yt, Wo) - np.dot(yto, Lo))/ Lodiag) - yto)


	metrics[e, :5] = (
		correlation_cost(W, xv, y),
		phi_capital(W, p, k),
		lateral_cost(L, Ldiag, p, q),
		np.linalg.norm(ff - y),
		np.linalg.norm(fb_ap - y)
	)
	metrics[e, -2] = np.mean(
		np.not_equal(
			np.argmax(yto, 1), 
			np.argmax(ytv, 1)
		)
	)
	metrics[e, -1] = cmds_cost(xv, y)
	Ch[e] = np.dot(xv.T, y)
	
	if e % 10 == 0:

		print "Epoch {}, {}".format(
			e,
			", ".join(["{:.4f}".format(m) for m in metrics[e, :]])
		)




from binding import *

seq_length = num_iters

c = NetConfig(
    Dt = dt,
    SeqLength=seq_length,
    BatchSize=batch_size,
    FeedbackDelay=1,
    OutputTau=5.0,
    YMeanStat = np.zeros((batch_size, seq_length, layer_size))
)

# W = np.random.randn(input_size, layer_size)
# W = W/(np.sum(np.abs(W), 0)/p)

net = (
    LayerConfig(
        Size = layer_size,
        TauSoma = 1.0,
        TauSyn = 1.0,
        TauSynFb = 1.0,
        TauMean = 1.0,
        P = p,
        Q = q,
        K = k,
        Omega = omega,
        FbFactor = 0.0,
        LearningRate=0.001,
        LateralLearnFactor=10.0,
        Act = RELU,
        W = Wcp, 
        B = np.ones((1, layer_size)),
        L = np.zeros((layer_size, layer_size)),
        dW = np.zeros((input_size, layer_size)),
        dB = np.zeros((1, layer_size)),
        dL = np.zeros((layer_size, layer_size)),
        Am = np.zeros((1, layer_size)),
        UStat = np.zeros((batch_size, seq_length, layer_size)),
        AStat = np.zeros((batch_size, seq_length, layer_size)),
        FbStat = np.zeros((batch_size, seq_length, layer_size)),
        SynStat = np.zeros((batch_size, seq_length, input_size)),
    ),
)

l0 = net[0]

Winit = l0.get("W")

fb_data = np.dot(yv, Wo.T)
t_fb_data = np.dot(ytv, Wo.T)


trainStats, testStats = run_model(
    10,
    net,
    c,
    xv,
    fb_data,
    xtv,
    t_fb_data,
    test_freq = 10
)

yh2 = l0.get("AStat").transpose((1,0,2))