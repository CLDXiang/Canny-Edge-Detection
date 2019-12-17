from PIL import Image
import numpy as np

def open_img(path):
    im = Image.open(path)
    if im.mode != 'RGB':
        print('[警告] 图片非RGB格式！')
    return im.convert('L')

def im2img(im):
    '''
    将 numpy 数组转为 PIL 的 Image 对象
    '''
    return Image.fromarray(im).convert('L')

def norm_im(im):
    im_norm = im * 256 / np.max(im)
    return im_norm

def save_im(im, path):
    im2img(im).save(path)

def im_binary(im):
    '''
    二值化
    '''
    im = im.copy()
    im[im > 0] = 255
    return im
