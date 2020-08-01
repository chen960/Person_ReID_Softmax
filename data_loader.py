from __future__ import absolute_import,print_function

import os
from IPython import embed
from PIL import Image
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset
# torch的dataloader叫dataset
# 本文件实现的是dataset功能，放在torch.dataloader中，并非dataloader
def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            # 本函数核心
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


# 数据库类三个必备函数  init len getitem
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)   # 这行出大问题  len（）忘记加了

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid




# if __name__ == '__main__':
#     import data_manager
#     dataset = data_manager.init_img_dataset(name='market1501', root='G:/dl_dataset')
#     # 等于dataset = Market1501（root='G:/dl_dataset')
#     train_loader = ImageDataset(dataset.train)
#     # for batch_id, (img,pid,camid) in enumerate(train_loader):
#     #     img.save('a.jpg')


