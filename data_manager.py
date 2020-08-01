from __future__ import print_function,absolute_import

import os
import os.path as osp
import numpy as np
from utils import mkdir_if_missing,write_json,read_json
import glob
import re    # regular expression  正则表达式
from IPython import embed
# 数据库几个重要参数  摄像头编号 行人编号（pid） 图片数量


class Market1501(object):
    dataset_dir = 'market1501'

    def __init__(self,root='data',**kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        #打印数据库信息
        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        #三个数据库
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
    # 检查上边路径函数

    def _check_before_run(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not avaluable".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not avaluable".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not avaluable".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not avaluable".format(self.gallery_dir))
    # 处理数据库  relabel用于训练阶段使用，将label重新按顺序排列，比如原来训练集id为 2 5 7....，重新排123
    # 由于是表征学习，在训练时按分类做，测试时去除最后FC层，如果不重新label，最终FC层需要1501个神经元
    # 重新relabel能够提高训练速度，减少最后FC层参数

    def _process_dir(self, dir_path, relabel=False):
        # 找出路径下的所有jpg格式文件，返回列表，内有所有文件名称字符串
        # 这里提取出pid 和 摄像头编号  摄像头编号用于在测试阶段剔除同摄像头图像
        img_paths = glob.glob(osp.join(dir_path,'*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        # pattern给定模式，search返回匹配对象  groups返魂所有匹配分组  map寻找整数匹配对象 即pid 和cid
        # 这部分用于获得重排序的label
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, re.search(pattern, img_path).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        # pid2label是一个字典，键值对为  pid：重排标签



        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, re.search(pattern, img_path).groups())
            # 可以忽略背景图片，这里为了使训练困难，不选择忽略
            # if pid == -1:
            #     continue
            assert -1 <= pid <= 1501
            assert 1 <= camid <= 6
            camid -= 1
            #重排序
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        num_pids = len(pid_container)
        # 这里len不采用img_paths是因为它其中包含了背景图片，在dataset中被忽略
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

"""Create dataset"""

__img_factory = {
    'market1501': Market1501,
}


def get_names():
    return __img_factory.keys()


def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)

# 调试使用
# if __name__ == '__main__':
#     data = Market1501(root='G:/dl_dataset')
