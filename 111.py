# a = "sadhfkjwhefkjnzkxchjksadfm"
# b = a.split("aaa")
# c = a.upper()
#
# import torch
# a = torch.tensor([1, 2, 3, 4, 5, 6, 7])
# b = torch.tensor([1, 2, 2, 3, 7, 6, 7])
# c = a == b
# c = c.to(dtype=torch.long)
#
# print([a == c for c in a])


# import os
# import logging
# from six.moves import urllib
# def _download_split_info():
#     """ this function aims to download the Omniglot split information (Vinyals et al. 2016) """
#     # if no split info. files, try to download
#     __ravi_url = \
#         'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/miniImagenet/splits/ravi/'
#     __ravi_protocol = {
#         'test': __ravi_url + 'test.csv',
#         'train': __ravi_url + 'train.csv',
#         'val': __ravi_url + 'val.csv',
#     }
#     files = os.listdir(os.getcwd())
#     for key, url in __ravi_protocol.items():
#         _file = key + '.csv'
#         if _file not in files:
#             logging.info("starting download %s info. from %s" % (key, url))
#             _data = urllib.request.urlopen(url)
#             _file = os.path.join(os.getcwd(), _file)
#             with open(_file, "wb") as f:
#                 f.write(_data.read())
#     logging.info("successfully download Omniglot split info. (Vinyals)")
#
#
# logging.basicConfig(level=logging.INFO,
#                     format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")
# # _download_split_info()
#
# import os
# import pandas as pd
# a = pd.read_csv(os.path.join(os.getcwd(), "train.csv"), sep=",").values
# import numpy as np
# classes = np.unique(a[:, 1])
# task = {cls:[] for cls in classes}
# for item, cls in a:
#     task[cls].append(item)
#
# cur_class = []
# with open(os.path.join(os.getcwd(), "train.csv"), "r") as f:
#     for cls in f.readlines():
#         cur_class.append(cls.strip())

from dataflow.utils import pil_rgb_loader
size = (84, 84)
f1 = "E:\\Transferring_Datasets\\Mini_ImageNet\\images\\n0425813800000644.jpg"
f2 = "E:\\Transferring_Datasets\\Mini_ImageNet\\n0425813800000644.jpg"
f3 = "E:\\Transferring_Datasets\\Omniglot\\processed_images\\Alphabet_of_the_Magi\\character01\\0709_01.png"
import time
for i in range(10):
    t1 = time.time()
    s1 = pil_rgb_loader(f1)
    t2 = time.time()
    s2 = pil_rgb_loader(f2)
    t3 = time.time()
    s3 = pil_rgb_loader(f3)
    t4 = time.time()
    print(t2-t1, "\t", t3-t2, "\t", t4-t3)


a = [1, 2, 3, 4, 5, 6, 7]
b = [0, 2, 1, 4, 3, 5]

print(a[b[:3]])