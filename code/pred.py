import matplotlib
matplotlib.use('Agg')

import matplotlib.pylab as plt
import os
import sys
import argparse
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path of the image')
parser.add_argument('--model', help='Path of the model')
parser.add_argument('--usegpu', action='store_true', default=True,
                    help='Enables cuda to predict on gpu')
parser.add_argument('--output', default='/media/snowday/Seagate Backup Plus Drive/秋葵深度学习/mask',
                    help='Path of the output directory')
parser.add_argument('--dataset', type=str, default='CVPPP',
                    help='Name of the dataset which is "CVPPP"',
                    )
opt = parser.parse_args()

assert opt.dataset in ['CVPPP', ]
opt.model = '/media/snowday/045A0A095A09F7E6/git/segmentation/instance-segmentation-atten/models/CVPPP/2019-4-5_14-7_snowday_39-8722/model_134_0.06270705163478851_0.25.pth'
opt.image = '/media/snowday/Seagate Backup Plus Drive/秋葵深度学习/VIS-IMage/1.jpg'
image_path = opt.image
model_path = opt.model
output_path = opt.output

try:
    os.makedirs(output_path)
except BaseException:
    pass

model_dir = os.path.dirname(model_path)
# sys.path.insert(0, model_dir)

from lib import Model, Prediction

if opt.dataset == 'CVPPP':
    from settings import CVPPPModelSettings
    ms = CVPPPModelSettings()

model = Model(opt.dataset, ms.MODEL_NAME, ms.N_CLASSES, ms.MAX_N_OBJECTS,wae_opt=ms.wae_opt,
              use_instance_segmentation=ms.USE_INSTANCE_SEGMENTATION, use_wae=True,
              use_coords=ms.USE_COORDINATES, load_model_path=opt.model,
              usegpu=opt.usegpu)

prediction = Prediction(ms.IMAGE_HEIGHT, ms.IMAGE_WIDTH,
                        ms.MEAN, ms.STD, False, model,
                        1)



def handleins():
    #semantic segmentation

    image, fg_seg_pred, ins_seg_pred, n_objects_pred = prediction.predict(
        image_path)

    fg_seg_pred_norm = fg_seg_pred * 255

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    image_pil = Image.fromarray(image)
    fg_seg_pred_pil = Image.fromarray(fg_seg_pred_norm).convert('P')
    image_pil.save(os.path.join(output_path, image_name + '.png'))
    fg_seg_pred_pil.save(os.path.join(output_path, image_name + '-fg_mask.png'))

    #handle instance
    _n_clusters = len(np.unique(ins_seg_pred.flatten())) - 1  # discard bg
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]
    ins_seg_pred_color = np.zeros(
        (ins_seg_pred.shape[0], ins_seg_pred.shape[1], 3), dtype=np.uint8)
    for i in range(_n_clusters):
        ins_seg_pred_color[ins_seg_pred == (
                i + 1)] = (np.array(colors[i][:3]) * 255).astype('int')

    ins_seg_pred_pil = Image.fromarray(ins_seg_pred)
    ins_seg_pred_color_pil = Image.fromarray(ins_seg_pred_color)

    ins_seg_pred_pil.save(os.path.join(output_path, image_name + '-ins_mask.png'))
    ins_seg_pred_color_pil.save(os.path.join(
        output_path, image_name + '-ins_mask_color.png'))
    np.save(
        os.path.join(
            output_path,
            image_name +
            '-n_objects.npy'),
        n_objects_pred)

def genmask():
    root = '/media/snowday/Seagate Backup Plus Drive/秋葵深度学习/VIS-IMage'
    for fpath, dir, files in os.walk(root):
        for file in files:
            image_path = os.path.join(fpath, file)

            image, fg_seg_pred = prediction.predict(
                image_path)

            fg_seg_pred_norm = ((fg_seg_pred>0.5) * 255).astype(np.float32)
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # image_pil = Image.fromarray(image)
            # image_pil.save(os.path.join(output_path, image_name + '.png'))
            fg_seg_pred_pil = Image.fromarray(fg_seg_pred_norm).convert('P')
            fg_seg_pred_pil.save(os.path.join(output_path, image_name + '-fg_mask.png'))

if __name__ == '__main__':
    # genmask()
    image_path = '/media/snowday/045A0A095A09F7E6/envi/result_envi/0502_5_RT_rgb.png'

    image, fg_seg_pred = prediction.predict(
        image_path)

    fg_seg_pred_norm = ((fg_seg_pred > 0.5) * 255).astype(np.float32)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # image_pil = Image.fromarray(image)
    # image_pil.save(os.path.join(output_path, image_name + '.png'))
    fg_seg_pred_pil = Image.fromarray(fg_seg_pred_norm).convert('P')
    fg_seg_pred_pil.save(os.path.join(output_path, image_name + '-fg_mask.png'))
