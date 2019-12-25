# Canny-Edge-Detection
Implementation of Canny edge detection algorithem with pure python.

## Requirements

- [Numpy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)

## How to start

```shell
pip install numpy
pip install pillow
```

Change the parameters in "canny.py" into what you want. Then

```shell
python canny.py
```

And you can find the results in 'img/{img_name}'

## API

In `canny.py`:

**canny(img_path, img_name='default', k=2, sigma=1.4, fast_mode=False, auto_threshold=True, ratio=0.05, high=60., low=30.)**

- `img_path`: Path of your input image file.
- `img_name`: The directory name you want to store the results.
- `k`, `sigma`: Parameters of Gaussian filter.
- `fast_mode`: Whether to use |Gx + Gy| instead of sqrt(Gx^2 + Gy^2) when calculating the gradient which will speedup the calculation.
- `auto_threshold`: Whether to calculate the thresholds automatically.
- `ratio`: When auto_threshold is True, the ratio of strong edge pixels in all pixels.
- `high`, `low`: When auto_threshold is False, the high and low thresholds you want to set mannually.

**canny_base_on_NMS(img_nms_path, img_name='default', ratio=0.05)**

This function can be used when a output image of NMS exists. This is useful when you want to see the difference of results when diffrent `ratio` is choosed.

## 前置需求

- [Numpy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)

## 如何开始


```shell
pip install numpy
pip install pillow
```

按需改变 “canny.py” 中的参数，然后执行

```shell
python canny.py
```

结果会输出在 “img/{img_name}” 目录下。

## API

于 “canny.py” 中：

**canny(img_path, img_name='default', k=2, sigma=1.4, fast_mode=False, auto_threshold=True, ratio=0.05, high=60., low=30.)**

- `img_path`: 输入图片路径
- `img_name`: 图片名称（用于保存输出，不要带后缀）
- `k`: 高斯滤波器半径
- `sigma`: 高斯滤波器参数
- `fast_mode`: 在求梯度时是否采用近似估算以加速
- `auto_threshold`: 是否自动计算阈值
- `ratio`: 求阈值时强边缘占所有像素的比例（越大输出边缘越密集）
- `high`: 不自动计算阈值时的高阈值
- `low`: 不自动计算阈值时的低阈值

**canny_base_on_NMS(img_nms_path, img_name='default', ratio=0.05)**

在已经有 NMS 这一步得到的输出时使用，可以略过前面的步骤直接从 NMS 之后开始运行。用于对 `ratio` 进行调参。
