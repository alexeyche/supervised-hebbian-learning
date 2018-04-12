
import numpy as np
import ctypes as ct
import time
from binding import *
from util import *
from datasets import *
from sklearn.metrics import log_loss
from poc.common import *
from poc.opt import *
from sklearn import datasets as sklearn_datasets
from sklearn.feature_extraction.image import extract_patches_2d

np.random.seed(12)
rng = np.random.RandomState(12)

faces = sklearn_datasets.fetch_olivetti_faces()
image_num = faces.images.shape[0]
image_num = 20  # TODO

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

data_size, input_size = data.shape
data_size, output_size = output.shape



batch_size = 100
seq_length = 50
layer_size = 50

t_data = data[:batch_size].copy()
t_output = output[:batch_size].copy()


p = 0.1
q = 0.9
k = 1.0
omega = 1.0

Wo = positive_random_norm(output_size, layer_size, p)
fb_data = np.dot(output, Wo)
t_fb_data = np.dot(t_output, Wo)


c = NetConfig(
    Dt = 1.0,
    SeqLength=seq_length,
    BatchSize=batch_size,
    FeedbackDelay=1,
    OutputTau=5.0,
    YMeanStat = np.zeros((batch_size, seq_length, layer_size))
)

# W = np.random.randn(input_size, layer_size)
# W = W/(np.sum(np.abs(W), 0)/p)
W = positive_random_norm(input_size, layer_size, p)

net = (
    LayerConfig(
        Size = layer_size,
        TauSoma = 5.0,
        TauSyn = 15.0,
        TauSynFb = 5.0,
        TauMean = 100.0,
        P = p,
        Q = q,
        K = k,
        Omega = omega,
        FbFactor = 0.0,
        LearningRate=1e-06,
        LateralLearnFactor=10.0,
        Act = RELU,
        W = W, 
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


trainStats, testStats = run_model(
    20,
    net,
    c,
    data,
    fb_data,
    t_data,
    t_fb_data,
    test_freq = 10
)


st = trainStats
SquaredError = st.get("SquaredError")
ClassificationError = st.get("ClassificationError")
SignAgreement = st.get("SignAgreement")
AverageActivity = st.get("AverageActivity")
Sparsity = st.get("Sparsity")

shl(l0.get("AStat")[0,:])


## need metrics
## need good data to test

## http://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html#sphx-glr-auto-examples-cluster-plot-dict-face-patches-py
