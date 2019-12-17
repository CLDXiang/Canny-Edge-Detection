import numpy as np


def calcu_threshold(G, ratio=0.3):
    '''
    自动生成阈值，以前 30%(ratio) 为 high， high/2 为 low
    '''
    high = np.sort(G.flat)[int(G.size * (1 - ratio))]
    low = high / 2
    return high, low


def two_threshold(G, high, low):
    '''
    双阈值算法，以 2 表示强边缘， 1 表示弱边缘， 0 表示应该被抑制
    '''
    G_class = np.zeros(G.shape)

    last_notice = -1  # 最后一次报告的进度
    print('正在进行双阈值检测...')
    for i in range(1, G.shape[0] - 1):
        process = int(i * 100 / G.shape[0])  # 进度百分比
        if process % 10 == 0 and process > last_notice:
            print('双阈值检测进度： {}%'.format(process))
            last_notice = process
        for j in range(1, G.shape[1] - 1):

            if G[i, j] >= high:
                G_class[i, j] = 2
            elif G[i, j] >= low:
                G_class[i, j] = 1

    print('双阈值检测完成')

    return G_class


def depress_outliers(G, G_class):
    '''
    抑制孤立的弱边缘
    '''

    # 1
    s = []
    q = []
    connected = False
    mark = np.zeros(G.shape)  # 标记矩阵

    last_notice = -1  # 最后一次报告的进度
    print('正在进行孤立弱边缘抑制...')
    for i in range(1, G.shape[0] - 1):
        process = int(i * 100 / G.shape[0])  # 进度百分比
        if process % 10 == 0 and process > last_notice:
            print('孤立弱边缘抑制进度： {}%'.format(process))
            last_notice = process
        for j in range(1, G.shape[1] - 1):

            # 2
            if G_class[i, j] == 1 and mark[i, j] == 0:
                mark[i, j] = 1  # 标记
                s.append((i, j))  # 入栈
                q.append((i, j))  # 入队
            else:
                continue  # 进入下一个点

            # 3
            while len(s):
                i_, j_ = s.pop()  # 从栈中取出一个元素
                # 扫描相邻像素
                for y_bias in range(-1, 2):
                    for x_bias in range(-1, 2):
                        if x_bias == 0 and y_bias == 0:
                            continue

                        i_a, j_a = i_ + y_bias, j_ + x_bias  # 相邻像素索引值

                        if G_class[i_a, j_a] == 1 and mark[i_a, j_a] == 0:
                            mark[i_a, j_a] = 1  # 标记
                            s.append((i_a, j_a))  # 入栈
                            q.append((i_a, j_a))  # 入队
                        elif G_class[i_a, j_a] == 2:  # 若和强边缘相邻
                            connected = True

            if not connected:
                for i_a, j_a in q:
                    mark[i_a, j_a] = 0  # 取消标记

            q = []
            connected = False

    G_res = G.copy()

    print('正在渲染结果...')
    for i in range(1, G.shape[0] - 1):
        for j in range(1, G.shape[1] - 1):
            if G_class[i, j] != 2 and mark[i, j] != 1:  # 既不是强边缘也没有被标记
                G_res[i, j] = 0

    print('孤立弱边缘抑制完成')

    return G_res
