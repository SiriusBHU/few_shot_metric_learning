import numpy as np

a = np.load("E:\\Transferring_Datasets\\Tiered_Imagenet\\few-shot-train.npz")
print(1)

b = a['features']


import pickle
with open("E:\\Transferring_Datasets\\tiered-imagenet\\tiered-imagenet\\train_images_png.pkl", 'rb') as f:
    a = pickle.load(f)

with open("E:\\Transferring_Datasets\\tiered-imagenet\\tiered-imagenet\\train_labels.pkl", 'rb') as f:
    b = pickle.load(f)

from tqdm import tqdm
import cv2

kk1 = []
aa = iter(tqdm(enumerate(a)))
for i in range(10):
    ii, item = aa.__next__()
    kkkkk = cv2.imdecode(item, 1)
    kk1.append(kkkkk)
print(1)

with open("a1.pkl", "wb") as f:
    pickle.dump(kk1[5], f)


category = np.unique(b["label_general"])
classes = np.unique(b["label_specific"])

ct_of_cls = [b["label_general"][c == b["label_specific"]] for c in classes]
ct_of_cls = [np.unique(item) for item in ct_of_cls]
ct_of_cls = np.array(ct_of_cls)


def process_pkl2png():
    pass



import os
import pickle
directory = 'E:\\Transferring_Datasets\\CIFAR100'
with open(os.path.join(directory, 'meta'), 'rb') as fi:
    meta = pickle.load(fi, encoding='latin1')

fine_classes = {cls: idx for idx, cls in enumerate(meta['fine_label_names'])}
coarse_classes = {cls: idx for idx, cls in enumerate(meta['coarse_label_names'])}


split_mode = ('train', 'val', 'test')
protocol = {mode: [] for mode in split_mode}
directory_split = 'E:\\Transferring_Datasets\\CIFAR100\\bertinetto'
for mode in split_mode:
    with open(os.path.join(directory_split, mode + '.txt'), 'r') as ff:
        for cls in ff.readlines():
            cls = cls.strip()
            protocol[mode].append(fine_classes[cls])
print(protocol)


import os
import pickle
directory = 'E:\\Transferring_Datasets\\CIFAR100'
with open(os.path.join(directory, 'train'), 'rb') as fi:
    entry = pickle.load(fi, encoding='latin1')

























