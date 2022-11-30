import pandas as pd, numpy as np, shutil, os, sys, pickle as pkl
shutil.copytree("./data/crossdock2020", "./data/crossdock2020_core", ignore=shutil.ignore_patterns('*.*'))
with open("/home/disk2/sf/Projects/GraphBP/GraphBP_20/trained_model/gen_mols_epoch_33/global_index_to_rec_src.dict", mode="rb") as f:
    rec_dict = pkl.load(f)
with open("/home/disk2/sf/Projects/GraphBP/GraphBP_20/trained_model/gen_mols_epoch_33/global_index_to_ref_lig_src.dict", mode="rb") as f:
    lig_dict = pkl.load(f)
# have duplicate, can use set(...) to avoid re-writing
for src in lig_dict.values():
    shutil.copy("./data/crossdock2020/"+src, "./data/crossdock2020_core/"+src)
for src in rec_dict.values():
    shutil.copy("./data/crossdock2020/"+src, "./data/crossdock2020_core/"+src)