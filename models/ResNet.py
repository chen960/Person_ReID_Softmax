from __future__ import absolute_import, print_function

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from IPython import embed
#网络模型都要继承 nn.module这个类
class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'softmax','metric'}, **kwargs):
        super(ResNet50,self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        # children和 list包装成列表  去除倒数两位 *加指针  用sequential
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        # torch.Size([32, 2048, 8, 4])  需要将其展平才能接全连接层  后边将每个图像特征池化为2048维向量
        x = F.avg_pool2d(x, x.size()[2:])
        # torch.Size([32, 2048, 1, 1])
        f = x.view(x.size()[0], -1)
        # torch.Size([32, 2048])
        # 归一化示例  torch.norm(f, 2, dim=-1, keepdim=True)算了模的平方
        # f = 1.*f /(torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f
        y = self.classifier(f)
        return y




# if __name__ == '__main__':
#     model = ResNet50(num_classes=751)
#     imgs = torch.Tensor(32, 3, 256, 128)
#     features = model(imgs)
#     embed()

# __factory = {
#     'resnet50': ResNet50,
# }
#
# def get_names():
#     return __factory.keys()
#
# def init_model(name, *args, **kwargs):
#     if name not in __factory.keys():
#         raise KeyError("Unknown model: {}".format(name))
#     return __factory[name](*args, **kwargs)