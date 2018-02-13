
import numpy as np

from shl import Config, LayerConfig
from shl import run_model

l0 = LayerConfig()
l0.layer_size = 4


l1 = LayerConfig()
l1.layer_size = 100


l2 = LayerConfig()
l2.layer_size = 2

c = Config()
c.layer_configs = (LayerConfig * 3)(l0, l1, l2)
c.layer_num = 3




run_model(
	c,
	np.random.random((10, 10))
)