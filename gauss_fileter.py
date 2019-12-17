import numpy as np
from PIL import Image
from math import pi, exp
import utils


def calculate_H(i, j, k, sigma=1.4):
    # 注意此处的 i, j 从 1 开始
    sigma_2 = sigma * sigma
    return 0.5 / pi / sigma_2 * exp(-((i - k - 1) ** 2 + (j - k - 1) ** 2) / (2 * sigma_2))


def gauss_filter(k=2, sigma=1.4):
    '''
    返回一个 (2k+1) * (2k+1) 的高斯核
    '''
    l = 2 * k + 1
    GF = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            GF[i, j] = calculate_H(i + 1, j + 1, k, sigma)
    return GF / np.sum(GF)


def gauss_smooth(img, k=2, sigma=1.4):
    '''
    高斯滤波
    '''
    im = np.array(img, dtype=np.float64)  # 转为nparr

    GF = gauss_filter(k, sigma)

    im_new = np.zeros(im.shape)

    last_notice = -1  # 最后一次报告的进度
    print('正在进行高斯模糊...')
    for i in range(im.shape[0]):
        process = int(i * 100 / im.shape[0])  # 进度百分比
        if process % 10 == 0 and process > last_notice:
            print('高斯模糊进度： {}%'.format(process))
            last_notice = process
        for j in range(im.shape[1]):

            if i < k or j < k or i >= im.shape[0] - k or j >= im.shape[1] - k:  # 边缘
                im_new[i, j] = im[i, j]  # 直接复制过来，或许也可以取个局部均值？
            else:
                im_new[i, j] = np.sum(im[i - k:i + k + 1, j - k:j + k + 1] * GF)

    print('高斯模糊完成')

    return im_new
