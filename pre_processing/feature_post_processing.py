import glob
import os
import pickle as pk

import h5py
import numpy as np
from tqdm import tqdm


def get_feature_list(root='~/datasets/', dataset='vcdb', feat='imac'):
    if dataset == 'vcdb':
        dataset = 'vcdb'
    elif dataset == 'ccweb':
        dataset = 'CC_WEB_VIDEO'
    elif dataset == 'fivr':
        dataset = 'FIVR-200K'
    if dataset == 'vcdb':
        return sorted(glob.glob(root + dataset + '/features/' + feat +  '/core/*.npy')) + sorted(glob.glob(root + dataset + '/features/' + feat +  '/[123456789]*/*.npy'))
    return sorted(glob.glob(root + dataset + '/features/' + feat +  '/*/*.npy'))

def export_feature_list(feature_list, out_path):
    with open(out_path, 'w') as f:
        for path in feature_list:
            f.write(path.split('/')[-1].split('.')[-2] + '\t' + path + '\n')

def npy2h5py(feature_list_path, h5path, pca=None):
    paths = [l.split('\t')[1].strip() for l in open(feature_list_path, 'r').readlines()]
    with h5py.File(h5path, 'w') as f:
        for path in tqdm(paths):
            vid = path.split('/')[-1].split('.')[-2]
            if pca:
                f.create_dataset(vid, data=pca.infer(np.load_path))
            else:
                f.create_dataset(vid, data=np.load(path))   

if __name__ == "__main__":
    feature_list = get_feature_list(dataset='vcdb', feat='imac')
    export_feature_list(feature_list, out_path='vcdb_feature_paths_imac.txt')
