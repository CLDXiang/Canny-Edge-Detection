from gauss_fileter import gauss_smooth
from sobel import get_gradient_and_direction
from NMS import *
from two_threshold import *
from utils import *
from os import path, mkdir


def canny(img_path, img_name='default', k=2, sigma=1.4, fast_mode=False, auto_threshold=True, ratio=0.05, high=60., low=30.):
    '''
    Canny 边缘检测算法
    :参数 img_path: 输入图片路径
    :参数 img_name: 图片名词（用于保存输出，不要带后缀）
    :参数 k: 高斯滤波器半径
    :参数 sigma: 高斯滤波器参数
    :参数 fast_mode: 在求梯度时是否采用近似估算以加速
    :参数 auto_threshold: 是否自动计算阈值
    :参数 ratio: 求阈值时强边缘占所有像素的比例（越大输出边缘越密集）
    :参数 high: 不自动计算阈值时的高阈值
    :参数 low: 不自动计算阈值时的低阈值
    '''
    img = open_img(img_path)

    if not path.exists('img'):
        mkdir('img')
    if not path.exists('img/{}'.format(img_name)):
        mkdir('img/{}'.format(img_name))

    img.save('img/{}/{}_gray.jpg'.format(img_name, img_name))

    # 高斯模糊
    im_gs = gauss_smooth(img, k, sigma)
    save_im(im_gs, 'img/{}/{}_gs.jpg'.format(img_name, img_name))

    # 计算梯度和方向
    G, theta = get_gradient_and_direction(im_gs, fast_mode)
    G_norm = norm_im(G)  # 正规化
    save_im(G_norm, 'img/{}/{}_gred.jpg'.format(img_name, img_name))

    # 非极大值抑制
    G_nms = NMS(G_norm, theta)
    save_im(G_nms, 'img/{}/{}_nms.jpg'.format(img_name, img_name))

    # 两阈值检测
    if auto_threshold:
        high, low = calcu_threshold(G_nms, ratio)  # 自动计算阈值
    G_class = two_threshold(G_nms, high, low)  # 分类：强边缘、弱边缘、无边缘
    G_res = depress_outliers(G_nms, G_class)  # 抑制孤立弱边缘
    save_im(G_res, 'img/{}/{}_res.jpg'.format(img_name, img_name))

    # 二值化
    G_bin = im_binary(G_res)
    save_im(G_bin, 'img/{}/{}_bin.jpg'.format(img_name, img_name))


def canny_base_on_NMS(img_nms_path, img_name='default', ratio=0.05):
    '''
    调参用，从 NMS 之后开始
    '''
    img = open_img(img_nms_path)
    G_nms = np.array(img, dtype=np.float64)

    if not path.exists('img'):
        mkdir('img')
    if not path.exists('img/{}'.format(img_name)):
        mkdir('img/{}'.format(img_name))

    # 两阈值检测
    high, low = calcu_threshold(G_nms, ratio)  # 自动计算阈值
    G_class = two_threshold(G_nms, high, low)  # 分类：强边缘、弱边缘、无边缘
    G_res = depress_outliers(G_nms, G_class)  # 抑制孤立弱边缘
    save_im(G_res, 'img/{}/{}_res.jpg'.format(img_name, img_name))

    # 二值化
    G_bin = im_binary(G_res)
    save_im(G_bin, 'img/{}/{}_bin.jpg'.format(img_name, img_name))


if __name__ == '__main__':
    # canny('img/lena_std.tif', 'lena_std')
    # canny('img/test1.jpg', 'test1')
    # canny_base_on_NMS('img/test1/test1_nms.tif', 'test1_nms', 0.02)
    # canny_base_on_NMS('img/lena_std/lena_std_nms.tif', 'lena_std_nms', 0.04)
    # for i in range(8, 10):
    #     canny('img/lena_std.tif', 'lena_std_k_1_ratio_0.0{}'.format(i), k = 1, ratio=0.01*i)


    # for i in [1, 2, 3, 5, 10, 20, 50, 100 ]:
    #     canny('img/lena_std.tif', 'lena_std_k_{}_h_25_4'.format(i), k=i, auto_threshold=False, high=25.4, low=12.7)

    for i in [1, 1.4, 2]:
        canny('img/lena_std.tif', 'lena_std_sigma_{}'.format(i), sigma=i)





