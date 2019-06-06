import os
import numpy as np
import random

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
                                        os.path.pardir, os.path.pardir))
IMG_DIR = os.path.join(DATA_DIR, 'raw')
METADATA_OUTPUT_DIR = os.path.join(DATA_DIR, 'metadata')

SUBSETS = ['train', 'val']
SUBSET_NAMES= ['training', 'validation']

def generate_lst(path):
    for _, _, files in os.walk(path):
        f = files
    f = list(filter(lambda x:'_rgb' in x, f))
    f = list(map(lambda x: x.split('_rgb')[0], f))
    random.shuffle(f)
    l = len(f)
    train_lst = os.path.join(METADATA_OUTPUT_DIR, SUBSET_NAMES[0] + '.lst')
    val_lst = os.path.join(METADATA_OUTPUT_DIR, SUBSET_NAMES[1] + '.lst')
    with open(train_lst, 'w') as file:
        for i in range(int(0.8*l)):
            file.write(f[i])
            file.write('\n')
    with open(val_lst, 'w') as file:
        for i in range(int(0.8*l), l):
            file.write(f[i])
            file.write('\n')
if not os.path.exists(os.path.join(METADATA_OUTPUT_DIR, SUBSET_NAMES[0] + '.lst')):
    generate_lst(IMG_DIR)

for si, subset in enumerate(SUBSETS):
    lst = os.path.join(METADATA_OUTPUT_DIR, SUBSET_NAMES[si] + '.lst')
    image_names = np.loadtxt(lst, dtype='str', delimiter=',')

    image_paths = []
    for image_name in image_names:
        _dir = image_name.split('_')[0]
        image_path = os.path.join(IMG_DIR, image_name + '_rgb.png')
        image_paths.append(image_path)

    np.savetxt(os.path.join(METADATA_OUTPUT_DIR, SUBSET_NAMES[si] + '_image_paths.txt'),
               image_paths, fmt='%s', delimiter=',')
