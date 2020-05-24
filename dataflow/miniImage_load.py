"""
    mini-ImageNet Data-Set ReadFile
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
import pandas as pd
import numpy as np
import os
import shutil
from six.moves import urllib
from torchvision.datasets.vision import VisionDataset
from dataflow.utils import make_taskset, pil_rgb_loader


class MiniImageNet(VisionDataset):

    r"""
        Arguments:
            path_images (string, optional): the directory store MiniImageNet data.
            mode (str, None, optional): within ['train', 'val', 'test'],
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
    __img_urls = "https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk"
    __ravi_url = \
        'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/miniImagenet/splits/ravi/'
    __ravi_protocol = {
        'test': __ravi_url + 'test.csv',
        'train': __ravi_url + 'train.csv',
        'val': __ravi_url + 'val.csv',
    }

    def __init__(self,
                 path_images=None,
                 mode=None,
                 loader=pil_rgb_loader,  # default_loader,
                 transform=None,
                 target_transform=None):

        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")

        # set image directory
        self.path_images = path_images
        if self.path_images is None:
            self.path_images = os.path.join(os.getcwd(), "dataset\\miniImageNet")
        # check data-set related directories do or not exist
        if not os.path.exists(self.path_images):
            logging.warning("no original mini-ImageNet image path")
            os.makedirs(self.path_images)

        # ------------------------------------------
        # check the images have or not been prepared
        # if not, prepare images
        self.path_processed = os.path.join(self.path_images, 'images')
        if not os.path.exists(self.path_processed) \
                or not len(os.listdir(self.path_processed)):
            raise FileExistsError("images of mini-ImageNet should be download first\n"
                                  "then please unzip images into directory '%s' .\n"
                                  "reference download url: %s"
                                  % (self.path_processed, self.__img_urls))

        # -------------------------------------
        # check the train-validation-test split
        # information has or hasn't been prepared
        # if not, prepare
        self.path_split = os.path.join(self.path_images, "ravi_split")
        # check the split information directories do or not exist
        if not os.path.exists(self.path_split):
            logging.warning("no mini-ImageNet ravi-split info. path")
            os.makedirs(self.path_split)
        # check the split information files do or not exist,
        # if no files, download
        if len(os.listdir(self.path_split)) < 3:
            logging.warning("no mini-ImageNet ravi-split info. files")
            self._download_split_info()

        # --------------------------------------
        # after guaranteeing the images has been
        # prepared in the setting path_images,
        # set root and declare its father class
        self.root = self.path_processed
        super(MiniImageNet, self).__init__(self.path_processed,
                                           transform=transform,
                                           target_transform=target_transform)

        # ---------------
        # prepare dataset

        # check whether load train, validation, or test task set
        if mode is None:
            logging.info("default loading training task set")
            mode = 'train'
        if mode not in ['train', 'val', 'test', 'trainval']:
            raise ValueError("expected data-set mode should within choices of "
                             "['train', 'val'(validation), 'trainval'(train + validation), 'test'], "
                             "but got {} instead\n".format(mode))

        # generate dataset
        classes, class_to_idx, samples = self._find_classes_items(mode, 'jpg')
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in sub-folders of: " + self.root + "\n"
                                "Supported file-type is *.png"))
        tasks = make_taskset(samples, class_to_idx)

        # -------------
        # set variables
        self.loader = loader
        self.mode = mode
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[-1] for s in samples]
        self.tasks = tasks
        # print self information
        logging.info(self)

    def _download_split_info(self):
        """ this function aims to download the miniImageNet split information (Ravi et al. 2017) """
        # if no split info. files, try to download
        files = os.listdir(self.path_split)
        for key, url in self.__ravi_protocol.items():
            _file = key + '.csv'
            if _file not in files:
                logging.info("starting download %s info. from %s" % (key, url))
                _data = urllib.request.urlopen(url)
                _file = os.path.join(self.path_split, _file)
                with open(_file, "wb") as f:
                    f.write(_data.read())
        logging.info("successfully download miniImageNet split info. (Sachin Ravi)")

    def _find_classes_items(self, mode, extensions='jpg'):

        # get ravi split info. file
        if mode != 'trainval':
            cur_mode_file = os.path.join(self.path_split, mode + '.csv')
            _items = pd.read_csv(cur_mode_file, sep=',').values
        else:
            _items = []
            for _mode in ('train', 'val'):
                _mode_file = os.path.join(self.path_split, _mode + '.csv')
                _items.append(pd.read_csv(_mode_file, sep=',').values)
            _items = np.concatenate(_items, axis=0)

        # get classes and the corresponding index of the current mode
        classes = list(np.unique(_items[:, -1]))
        classes.sort()
        class_to_idx = {_cls: i for i, _cls in enumerate(classes)}

        # formulate samples-set and the corresponding tasks-set
        images = []
        for _s, _cls in _items:
            _file = os.path.join(self.path_processed, _s)
            if self.__is_valid_file(_file, extensions):
                if os.path.exists(_file):
                    item = (_file, class_to_idx[_cls])
                    images.append(item)
                else:
                    continue
        return classes, class_to_idx, images

    @staticmethod
    def __is_valid_file(x, extensions):
        return x.lower().endswith(extensions)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):

        # cap = "=================================================================="
        head = self.mode.upper() + " DataSet " + self.__class__.__name__
        body = ["Root location: {}".format(self.path_processed),
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
    from dataflow.utils import TaskBatchSampler, TaskLoader
    import time

    path_cur = os.getcwd()
    path_img = "E:\\Transferring_Datasets\\Mini_ImageNet"   # path_cur + "/.."    "E:\\Transferring_Datasets\\Omniglot"
    mini_img_set = MiniImageNet(path_images=path_img,
                                mode="train",
                                transform=transforms.Compose([transforms.Resize((84, 84)),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                                   std=(0.229, 0.224, 0.225))]))
    mini_img_loader = TaskLoader(mini_img_set,
                                 iterations=100,
                                 n_way=5, k_shot=1, query_shot=5,
                                 batch_shuffle=True,
                                 task_shuffle=True,
                                 num_workers=2)
    t1 = time.time()
    for i, (s_samples, s_labels, q_samples, q_labels) in enumerate(mini_img_loader):
        t2 = time.time()
        print(i, ": ", t2 - t1)
        t1 = t2
