# 本文件用于对图像做数据transform 常见的transform pytorch中自带
# 这里就是示例写一个pytorch中没有的transform  随机采集图像的一块区域并resize到指定尺寸  或先放大图片再扣一个区域（本程序使用后一种）
# Random2DTransform
from __future__ import absolute_import, print_function

from torchvision.transforms import *
from PIL import Image
import random

# transform类必要两个函数：  __init__初始化   __call__实现主要功能


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    # interpolation是差值 resize时需要
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        # 随机数小于0.5直接resize
        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        # 大等于0.5则先resize放大再取一片区域，最终图片与<0.5一样大
        # 放大
        new_height, new_width = int(round(self.height*1.125)), int(round(self.width*1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        # 裁剪crop
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1+self.width, y1+self.height))

        return croped_img


if __name__ == '__main__':
    img = Image.open('G:/dl_dataset/market1501/bounding_box_train/0002_c1s1_000451_03.jpg')
    # transform = Random2DTranslation(256, 128, 0.5)
    transform = transforms.Compose(
        [
            Random2DTranslation(256, 128, 0.5),    #随机裁剪
            transforms.RandomHorizontalFlip(),     #随机水平翻转
            # transforms.ToTensor()
            # 转化为张量后就不能以图像格式显示

        ]
    )
    # 自动进行__call__
    img_t = transform(img)
    # print(img_t.shape)
    # 张量时才有shape

    import matplotlib.pyplot as plt

    plt.figure(12)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_t)
    plt.show()