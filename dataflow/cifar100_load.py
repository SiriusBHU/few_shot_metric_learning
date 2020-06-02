"""
    Omniglot Data-Set ReadFile
    Author: Sirius HU
    Created Date: 2020.04.26

    Data Introduction:
        ...

    for more details, refer to:
        url: ...
        publication: ...

    Function:
"""
from __future__ import print_function

import logging
import os
import pickle
import shutil
import tarfile

import numpy as np
from six.moves import urllib
from torchvision.datasets.vision import VisionDataset

from dataflow.utils import make_taskset, pil_array_to_image


class CIFAR100(VisionDataset):
    r"""
        Arguments:
            path_images (str, optional): the directory store Omniglot images.
            protocol (str, None, optional): within ['fc100', 'cifar_fs'],
                it decide to load which protocol as the few-shot scenario
                (default: None)
            mode (str, None, optional): within ['train', 'val', 'trainval', 'test'],
                it decide to load which files for data-loader (default: None).
            loader (callable): A function to load a sample given its path.
                (default: pil_grey_loader).
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it.
    """
    # define private values
    __img_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    __fc100_protocol = {
        'train': [1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19],
        'val': [8, 11, 13, 16],
        'trainval': [1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19, 8, 11, 13, 16],
        'test': [0, 7, 12, 14],
    }
    __cifar_fs_protocol = {
        'train': [90, 76, 93, 66, 79, 53, 17, 39, 22, 57,
                  35, 72, 31, 0, 1, 13, 51, 64, 80, 20,
                  85, 61, 97, 68, 50, 11, 74, 25, 82, 88,
                  18, 43, 81, 41, 92, 33, 30, 24, 3, 63,
                  6, 36, 45, 10, 16, 9, 91, 78, 12, 59,
                  75, 44, 28, 38, 52, 29, 65, 54, 96, 67,
                  56, 49, 37, 23],
        'val': [55, 48, 87, 40, 27, 73, 14, 4, 7, 89,
                32, 47, 15, 26, 71, 19],
        'trainval': [90, 76, 93, 66, 79, 53, 17, 39, 22, 57,
                     35, 72, 31, 0, 1, 13, 51, 64, 80, 20,
                     85, 61, 97, 68, 50, 11, 74, 25, 82, 88,
                     18, 43, 81, 41, 92, 33, 30, 24, 3, 63,
                     6, 36, 45, 10, 16, 9, 91, 78, 12, 59,
                     75, 44, 28, 38, 52, 29, 65, 54, 96, 67,
                     56, 49, 37, 23, 55, 48, 87, 40, 27, 73,
                     14, 4, 7, 89, 32, 47, 15, 26, 71, 19],
        'test': [2, 5, 8, 21, 34, 42, 46, 58, 60, 62,
                 69, 70, 77, 83, 84, 86, 94, 95, 98, 99]
    }

    def __init__(self,
                 path_images=None,
                 protocol=None,
                 mode=None,
                 loader=pil_array_to_image,
                 transform=None,
                 target_transform=None):

        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")

        # set image directory
        self.path_images = path_images
        if self.path_images is None:
            self.path_images = os.path.join(os.getcwd(), "dataset\\CIFAR100")
        # check the directory do or not exist
        if not os.path.exists(self.path_images):
            logging.warning("no original CIFAR100 image directory")
            os.makedirs(self.path_images)

        # ------------------------------------------
        # check the images have or not been prepared
        # if not, prepare images
        _files = os.listdir(self.path_images)
        if not len(_files) \
                or 'train' not in _files \
                or 'test' not in _files \
                or 'meta' not in _files:
            logging.warning("no CIFAR100 pickled images")
            self._download_images()

        # --------------------------------------
        # after guaranteeing the images has been
        # prepared in the setting path_images,
        # set root and declare its father class
        self.root = self.path_images
        super(CIFAR100, self).__init__(self.root,
                                       transform=transform,
                                       target_transform=target_transform)

        # ---------------
        # prepare dataset

        # check loading protocol
        if protocol is None:
            logging.info("default loading cifar_fs protocol")
            protocol = 'cifar_fs'
        if protocol not in ['fc100', 'cifar_fs']:
            raise ValueError("expected protocol should be within choices of "
                             "['fc100', 'cifar_fs'], "
                             "but got {} instead\n".format(protocol))
        # check loading mode
        # (whether to load train, validation, or test task set)
        if mode is None:
            logging.info("default loading training task set")
            mode = 'train'
        if mode not in ['train', 'val', 'trainval', 'test']:
            raise ValueError("expected dataset mode should within choices of "
                             "['train', 'val'(validation), 'trainval'(train + validation), 'test'], "
                             "but got {} instead\n".format(mode))

        # generate data-set
        data, targets, classes, class_to_idx = self._load_images_find_classes(protocol, mode)
        if len(data) == 0:
            raise RuntimeError("read pickled files error")
        tasks = make_taskset([(idx, cls) for idx, cls in enumerate(targets)],
                             class_to_idx)

        # -------------
        # set variables
        self.loader = loader
        self.protocol = protocol
        self.mode = mode
        # set dataset
        self.data = data
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.tasks = tasks
        # print self information
        logging.info(self)

    def _download_images(self):
        """
            this function aims to download the Omniglot Dataset,
            and unzip it
            after unzipping, delete the compressed file
        """
        # if no *.zip files, try to download
        files = os.listdir(self.path_images)
        if "cifar-100-python.tar.gz" not in files:
            logging.info("starting download CIFAR100 images")
            url = self.__img_url
            logging.info("--> downloading from %s" % url)
            _data = urllib.request.urlopen(url)
            _file = url.strip().split(os.altsep)[-1]
            _file = os.path.join(self.path_images, _file)
            with open(_file, "wb") as f:
                f.write(_data.read())
            logging.info("successfully download CIFAR100 package")

        # now try to unzip the compressed *.zip files
        # after unzip, delete the *.zip files
        logging.info("unzip files ...")
        _file = "cifar-100-python.tar.gz"
        _file = os.path.join(self.path_images, _file)
        with tarfile.open(_file, 'r:gz') as _ref:
            _ref.extractall(self.path_images)
        os.remove(_file)

        # move the extracted file to root directory
        _dir = os.path.join(self.path_images, _file.split('.')[0])
        for _f in os.listdir(_dir):
            shutil.move(os.path.join(_dir, _f), self.path_images)
        shutil.rmtree(_dir)

        logging.info("successfully unzip CIFAR100 images")
        return

    def _load_images_find_classes(self, protocol, mode):

        # load images
        data_files = ('train', 'test')
        images, fine_labels, coarse_labels = [], [], []
        for _f in data_files:
            with open(os.path.join(self.path_images, _f), 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                images.append(entry['data'])
                fine_labels.extend(entry['fine_labels'])
                coarse_labels.extend(entry['coarse_labels'])
        images = np.concatenate(images, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        fine_labels, coarse_labels = np.array(fine_labels), np.array(coarse_labels)

        # load classes information
        with open(os.path.join(self.path_images, 'meta'), 'rb') as fo:
            meta = pickle.load(fo, encoding='latin1')
            fine_classes = meta['fine_label_names']
            # coarse_classes = meta['coarse_label_names']

        # according to the protocol and mode,
        # extract the required class_index
        if protocol == 'cifar_fs':
            class_idx = self.__cifar_fs_protocol[mode]
        else:
            coarse_idx = self.__fc100_protocol[mode]
            _coordinates = [coarse_labels == c for c in coarse_idx]
            class_idx = np.unique(np.concatenate([fine_labels[_idx] for _idx in _coordinates], axis=0))

        # extract the corresponding data and targets
        data, targets, classes, class_to_idx = [], [], [], {}
        for new_idx, _idx in enumerate(class_idx):
            coord = fine_labels == _idx
            data.append(images[coord])
            targets.append(new_idx * np.ones((sum(coord),), dtype=np.int))
            classes.append(fine_classes[_idx])
            class_to_idx[fine_classes[_idx]] = new_idx
        data, targets = np.concatenate(data, axis=0), np.concatenate(targets, axis=0).tolist()

        return data, targets, classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.data[index], self.targets[index]
        sample = self.loader(sample)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):

        # cap = "=================================================================="
        head = self.mode.upper() + " DataSet " + self.__class__.__name__
        body = ["Root location: {}".format(self.root),
                "\t\tNumber of datapoints: {}".format(self.__len__()),
                "\t\tNumber of classes: {}".format(len(self.tasks))]
        s_nums = [len(self.tasks[idx]) for idx in range(len(self.tasks))]
        max_num, min_num = max(s_nums), min(s_nums)
        body += ["\t\tMaximum samples' num per class: %d" % max_num,
                 "\t\tMinimum samples' num per class: %d" % min_num,
                 self.transforms.__repr__(),
                 "\n"]

        return "\n" + head + "\n" + "\n".join(body)

    def clear_files(self):
        """
            considering the data have been polluted,
            clear all and re-download
        """
        shutil.rmtree(self.path_images)
        return


if __name__ == "__main__":

    from torchvision import transforms
    from dataflow.utils import TaskLoader

    path_cur = os.getcwd()
    path_img = "E:\\Transferring_Datasets\\CIFAR100"  # path_cur + "/.."    "E:\\Transferring_Datasets\\Omniglot"
    fc100_set = CIFAR100(path_images=path_img,
                         protocol='fc100',
                         mode="train",
                         transform=transforms.Compose([transforms.Resize((32, 32)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                            std=(0.229, 0.224, 0.225))]))
    fc100_loader = TaskLoader(fc100_set, n_way=5, k_shot=1, query_shot=5,
                              iterations=100,
                              batch_shuffle=True,
                              task_shuffle=True,
                              num_workers=2)

    import time

    t1 = time.time()
    for i, (s_samples, s_labels, q_samples, q_labels) in enumerate(fc100_loader):
        t2 = time.time()
        print(i, ": ", t2 - t1)
        t1 = t2
