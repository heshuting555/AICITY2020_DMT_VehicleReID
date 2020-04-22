import numpy as np
import os

base_dir = '/data/model/'

distmat =  np.load(base_dir + '0402_6/dist_mat.npy')
distmat += np.load(base_dir + '0407_1/dist_mat.npy')
distmat += np.load(base_dir + '0408_6/dist_mat.npy')

distmat += np.load(base_dir + '0409_2/dist_mat.npy')
distmat += np.load(base_dir + '0409_3/dist_mat.npy')

print('The shape of distmat is: {}'.format(distmat.shape))

sort_distmat_index = np.argsort(distmat, axis=1)

with open(os.path.join(base_dir, 'track2.txt'), 'w') as f:
    for item in sort_distmat_index:
        for i in range(99):
            f.write(str(item[i] + 1) + ' ')
        f.write(str(item[99] + 1) + '\n')
print('writing result to {}'.format(os.path.join(base_dir, 'track2.txt')))
