import os
import glob
import numpy as np
from PIL import Image

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
                                        os.path.pardir, os.path.pardir))
IMG_DIR = os.path.join(DATA_DIR, 'raw')

image_list = np.loadtxt(os.path.join(DATA_DIR, 'metadata',
                                     'training.lst'),
                        dtype='str', delimiter=',')

reds, greens, blues = [], [], []
for image_name in image_list:
    img = np.array(Image.open(os.path.join(IMG_DIR, image_name + '_rgb.png')))
    r, g, b = np.split(img, 3, axis=2)

    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    reds.extend(r)
    greens.extend(g)
    blues.extend(b)

reds = np.array(reds)
greens = np.array(greens)
blues = np.array(blues)

red_mean = np.mean(reds) / 255.
green_mean = np.mean(greens) / 255.
blue_mean = np.mean(blues) / 255.

red_std = np.std(reds) / 255.
green_std = np.std(greens) / 255.
blue_std = np.std(blues) / 255.

with open(os.path.join(DATA_DIR, 'metadata', 'means_and_stds.txt'), 'w') as fp:
    fp.write('RGB MEANS : {} {} {}\n'.format(red_mean, green_mean, blue_mean))
    fp.write('RGB STDS  : {} {} {}\n'.format(red_std, green_std, blue_std))
