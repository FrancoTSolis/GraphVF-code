import pandas as pd
import numpy as np
import os
import random

data_cols = ['low_rmsd', 'true_aff', 'xtal_rmsd', 'rec_src', 'lig_src', 'vina_aff']
train_file = 'it2_tt_0_lowrmsd_mols_train0_fixed.types'
train_lines = pd.read_csv(
                    train_file, sep=' ', names=data_cols, index_col=False
                            )
test_file = 'it2_tt_0_lowrmsd_mols_test0_fixed.types'
test_lines = pd.read_csv(
                    test_file, sep=' ', names=data_cols, index_col=False
                            )

def seed_fix(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_fix(1234)
train_lines_20 = train_lines.sample(frac=0.05)
test_lines_20 = test_lines.sample(frac=0.05)
train_lines_20.to_csv("crossdock2020/train_20.types", index=False, header=False, sep=' ')
test_lines_20.to_csv("crossdock2020/test_20.types", index=False, header=False, sep=' ')

