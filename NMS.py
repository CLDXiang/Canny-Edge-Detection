import numpy as np
from math import pi


def NMS(G, theta):
    '''
    非极大值抑制算法，输入梯度和方向矩阵
    '''
    G_new = np.zeros(G.shape)
    angle = theta * 180 / pi  # 换成角度制方便判断

    last_notice = -1  # 最后一次报告的进度
    print('正在进行非极大值抑制...')
    for i in range(1, G.shape[0] - 1):
        process = int(i * 100 / G.shape[0])  # 进度百分比
        if process % 10 == 0 and process > last_notice:
            print('非极大值抑制进度： {}%'.format(process))
            last_notice = process
        for j in range(1, G.shape[1] - 1):

            tan = np.tan(theta[i, j])
            if tan == 0:
                cot = np.inf
            else:
                cot = 1 / tan
            Gp1, Gp2 = 0, 0

            if 0 <= angle[i, j] < 45:  # 0 ~ 45 度
                Gp1 = tan * G[i, j + 1] + (1 - tan) * G[i - 1, j + 1]
                Gp2 = tan * G[i, j - 1] + (1 - tan) * G[i + 1, j - 1]
            elif 45 <= angle[i, j] <= 90:  # 45 ~ 90 度
                Gp1 = cot * G[i - 1, j] + (1 - cot) * G[i - 1, j + 1]
                Gp2 = cot * G[i + 1, j] + (1 - cot) * G[i + 1, j - 1]
            elif -90 <= angle[i, j] < -45:  # -90 ~ -45 度
                Gp1 = -tan * G[i + 1, j] + (1 + tan) * G[i + 1, j + 1]
                Gp2 = -tan * G[i - 1, j] + (1 + tan) * G[i - 1, j - 1]
            elif -45 <= angle[i, j] < 0:  # -45 ~ 0 度
                Gp1 = -cot * G[i, j + 1] + (1 + cot) * G[i + 1, j + 1]
                Gp2 = -cot * G[i, j - 1] + (1 + cot) * G[i - 1, j - 1]

            if G[i, j] >= Gp1 and G[i, j] >= Gp2:
                G_new[i, j] = G[i, j]
            else:
                G_new[i, j] = 0

    print('非极大值抑制完成')

    return G_new
