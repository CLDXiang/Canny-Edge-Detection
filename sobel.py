import numpy as np

Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def get_gradient_and_direction(im, fast_mode=False):
    '''
    通过 Sobel 算子得到梯度和方向，在 fast 模式下用近似方式计算梯度
    '''
    G = np.zeros(im.shape)
    theta = np.zeros(im.shape)

    last_notice = -1  # 最后一次报告的进度
    print('正在进行计算梯度和方向...')
    for i in range(1, im.shape[0] - 1):
        process = int(i * 100 / im.shape[0])  # 进度百分比
        if process % 10 == 0 and process > last_notice:
            print('梯度和方向计算进度： {}%'.format(process))
            last_notice = process
        for j in range(1, im.shape[1] - 1):

            Gx = np.sum(im[i - 1:i + 2, j - 1:j + 2] * Sx)
            Gy = np.sum(im[i - 1:i + 2, j - 1:j + 2] * Sy)
            if fast_mode:
                G[i, j] = np.abs(Gx) + np.abs(Gy)
            else:
                G[i, j] = np.sqrt(np.square(Gx) + np.square(Gy))

            if Gx == 0:
                theta[i, j] = np.arctan(np.inf)
            else:
                theta[i, j] = np.arctan(Gy / Gx)

    print('梯度和方向计算完成')

    return G, theta
