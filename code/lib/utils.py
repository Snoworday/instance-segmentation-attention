from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from io import StringIO
from skimage.color import (rgb2lab, rgb2yuv, rgb2yiq, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
                           rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)
from preprocess import RandomResizedCrop, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomTranspose, RandomRotate, \
    RandomChannelSwap, RandomGamma, RandomResolution, Standardization, CenterCut


class ImageUtilities(object):

    @staticmethod
    def read_image(image_path, is_raw=False):
        if is_raw:
            img = Image.open(StringIO(image_path))
        else:
            img = Image.open(image_path).convert('RGB')
        img_copy = img.copy()
        img.close()
        return img_copy

    @staticmethod
    def image_resizer(height, width, interpolation=Image.BILINEAR):
        return transforms.Resize((height, width), interpolation=interpolation)

    @staticmethod
    def image_random_cropper_and_resizer(height, width, interpolation=Image.BILINEAR):
        return RandomResizedCrop(height, width, interpolation=interpolation)

    @staticmethod
    def image_random_horizontal_flipper():
        return RandomHorizontalFlip()

    @staticmethod
    def image_random_vertical_flipper():
        return RandomVerticalFlip()

    @staticmethod
    def image_random_transposer():
        return RandomTranspose()

    @staticmethod
    def image_normalizer(mean, std):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    @staticmethod
    def image_random_rotator(interpolation=Image.BILINEAR, random_bg=True):
        return RandomRotate(interpolation=interpolation, random_bg=random_bg)

    @staticmethod
    def image_random_90x_rotator(interpolation=Image.BILINEAR):
        return RandomRotate(interpolation=interpolation, random_bg=False)

    @staticmethod
    def image_random_color_jitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2):
        return transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    @staticmethod
    def image_random_grayscaler(p=0.5):
        return transforms.RandomGrayscale(p=p)

    @staticmethod
    def image_random_channel_swapper(p=0.5):
        return RandomChannelSwap(prob=p)

    @staticmethod
    def image_random_gamma(gamma_range, gain=1):
        return RandomGamma(gamma_range, gain=gain)

    @staticmethod
    def image_random_resolution(ratio_range):
        return RandomResolution(ratio_range)

    @staticmethod
    def image_standarder():
        return transforms.Compose([transforms.ToTensor(), Standardization()])

    @staticmethod
    def image_ex_standarder():
        return transforms.Compose([ImageEx(), transforms.ToTensor(), Standardization()])

    @staticmethod
    def image_center_cut():
        return CenterCut()


class ImageEx(object):

    def __call__(self, pic):
        """
        Args:
            Tensor: Converted image. # pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        rgb = np.asarray(pic)

        lab = rgb2lab(rgb)
        hsv = rgb2hsv(rgb)
        yuv = rgb2yuv(rgb)
        ycbcr = rgb2ycbcr(rgb)
        hed = rgb2hed(rgb)
        yiq = rgb2yiq(rgb)

        all = np.concatenate([rgb, lab, hsv, yuv, ycbcr, hed, yiq], axis=2).astype(np.float32)
        # all = all.transpose(2, 0, 1)
        # ciexyz = self.toCIE(pic)
        # cmyk = self.toCMYK(pic)
        return all

    def __repr__(self):
        return self.__class__.__name__ + '()'

    # M = torch.FloatTensor([[0.4124, 0.3576, 0.1805],
    #                        [0.2126, 0.7152, 0.0722],
    #                        [0.0193, 0.1192, 0.9505]])
    # N = torch.FloatTensor([95.047, 100, 108.883]).unsqueeze(1)
    # def toLAB(self, rgb):
    #     xb = ( (rgb+0.055)/ 1.055) ** 2.4
    #     xm = rgb / 12.92
    #     gammax = torch.where( rgb > 0.04045, xb, xm)
    #     c, h, w = gammax.shape
    #     xyz = torch.mm(M, gammax.view(b, -1))
    #     xyz_n = xyz / N
    #     f = torch.where(
    #         xyz_n > (6/29)**3,
    #         xyz_n ** 1/3,
    #         1/3*(29/6)**2*xyz_n + 4/29
    #     )
    #     l = (116 * f[1] - 16).unsqueeze(0)
    #     a = 500 * (f[0] - f[1]).unsqueeze(0)
    #     b = 200 * (f[1] - f[2]).unsqueeze(0)
    #     lab = torch.cat([l, a, b]).view(3, h, w)
    #     return lab
    # def toHSV(self, rgb):
    #     Cmax, index = torch.max(rgb, dim=0)
    #     Cmin = torch.min(rgb, dim=0)[0]
    #     delta = Cmax - Cmin
    #
    #     H0 = 0
    #     H1 =



import cv2
def onehot2idx(Index):
    onehot = list(filter(lambda x: x is not None, [idx if _hasp else None for idx, _hasp in enumerate(Index)]))
    return onehot

def writeProJpg(prob, background, name, point=None):
    back = background[1].view(256, 256)
    pro = prob[1].view(256, 256)
    Min = torch.min(pro.masked_fill( (1-back).byte(), np.inf ))
    Max = torch.max(pro.masked_fill( (1-back).byte(), -np.inf ))
    pro = (pro - Min) / (Max - Min)
    pro = pro.masked_fill( (1-back).byte(), 0)
    m = pro.view(256, 256, 1).cpu().detach().numpy()
    m = (np.concatenate([m, m, m], axis=2) * 255).astype(np.uint8)
    if point is not None:
        m[point[1]//256, point[1]%256, 0] = 0
        m[point[1]//256, point[1]%256, 1] = 0
        m[point[1]//256, point[1]%256, 2] = 255
    cv2.imwrite(name, m)

def writePnJpg(p_n, background):
    p_n = (p_n[1]>0.5).view(256, 256, 1).float().cpu().detach().numpy()
    back = background[1].view(256, 256, 1).cpu().detach().numpy() * 255
    p_n = p_n * back
    m = (np.concatenate([back, back, p_n], axis=2)).astype(np.uint8)
    cv2.imwrite('p_n.jpg', m)
# import cv2
# img = torch.FloatTensor(semantic_annotations[0])
# h,w = img.shape
# p_n = (img>0.5).view(h , w, 1).float().cpu().detach().numpy()*255
# m = (np.concatenate([p_n, p_n, p_n], axis=2)).astype(np.uint8)
# cv2.imwrite('p_n.jpg', m)