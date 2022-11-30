from config import conf
from runner import Runner
import os, random, torch, numpy as np
import warnings
from numba import njit
warnings.filterwarnings("ignore")


binding_site_range = 15.0

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@njit
def set_seed(value):
    random.seed(1234)
    np.random.seed(value)

set_seed(1234)
seed_torch(1234)

out_path = 'trained_model'
if not os.path.isdir(out_path):
    os.mkdir(out_path)

runner = Runner(conf, out_path=out_path)
runner.train(binding_site_range)
