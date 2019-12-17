from gauss_fileter import gauss_smooth
from sobel import get_gradient_and_direction
from NMS import *
from two_threshold import *
from utils import *
from os import path, mkdir

def canny(img_path, img_name='default', k=2, sigma=1.4, fast_mode=False, ratio=0.05):
    '''
    img_name 不需要包括后缀！
    '''
    img = open_img(img_path)

    if not path.exists('img'):
        mkdir('img')
    if not path.exists('img/{}'.format(img_name)):
        mkdir('img/{}'.format(img_name))

    img.save('img/{}/{}_origin.tif'.format(img_name, img_name))

    # 高斯模糊
    im_gs = gauss_smooth(img, k, sigma)
    save_im(im_gs, 'img/{}/{}_gs.tif'.format(img_name, img_name))

    # 计算梯度和方向
    G, theta = get_gradient_and_direction(im_gs, fast_mode)
    G_norm = norm_im(G) # 正规化
    save_im(G_norm, 'img/{}/{}_gred.tif'.format(img_name, img_name))

    # 非极大值抑制
    G_nms = NMS(G_norm, theta)
    save_im(G_nms, 'img/{}/{}_nms.tif'.format(img_name, img_name))

    # 两阈值检测
    high, low = calcu_threshold(G_nms, ratio) # 自动计算阈值
    G_class = two_threshold(G_nms, high, low) # 分类：强边缘、弱边缘、无边缘
    G_res = depress_outliers(G_nms, G_class) # 抑制孤立弱边缘
    save_im(G_res, 'img/{}/{}_res.tif'.format(img_name, img_name))

    # 二值化
    G_bin = im_binary(G_res)
    save_im(G_bin, 'img/{}/{}_bin.tif'.format(img_name, img_name))

if __name__ == '__main__':
    # canny('img/lena_std.tif', 'lena_std')
    canny('img/test1.jpg', 'test1')





