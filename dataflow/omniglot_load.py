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
import shutil
import zipfile
from six.moves import urllib
from torchvision.datasets.vision import VisionDataset
from dataflow.utils import make_taskset, pil_grey_loader


class OmniglotVinyals(VisionDataset):

    r"""
        Arguments:
            path_images (string, optional): the directory store Omniglot images.
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
    __img_urls = ["https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
                  "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"]
    __vinalys_url = \
        'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    __vinyals_protocol = {
        'test': __vinalys_url + 'test.txt',
        'train': __vinalys_url + 'train.txt',
        'trainval': __vinalys_url + 'trainval.txt',
        'val': __vinalys_url + 'val.txt',
    }
    __trans_opts = (0, 90, 180, 270)

    def __init__(self,
                 path_images=None,
                 mode=None,
                 loader=pil_grey_loader,  # default_loader,
                 transform=None,
                 target_transform=None):

        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")

        # check image path
        self.path_images = path_images
        if self.path_images is None:
            self.path_images = os.path.join(os.getcwd(), "dataset\\Omniglot")
        # check data-set related directories do or not exist
        if not os.path.exists(self.path_images):
            logging.warning("no original Omniglot image path")
            os.makedirs(self.path_images)

        # check the data-set has or hasn't been prepared
        # if not, prepare dataset
        self.path_processed = os.path.join(self.path_images, "processed_images")
        # check the processed images directories do or not exist
        if not os.path.exists(self.path_processed):
            logging.warning("no processed Omniglot image path")
            os.makedirs(self.path_processed)
        # check the processed files do or not exist,
        # if no files, process origin image into the processed-path
        if len(os.listdir(self.path_processed)) < 50:
            logging.warning("no processed Omniglot image files")
            # check if there has original images,
            # if no images, download and unzip them
            files = os.listdir(self.path_images)
            if "images_background" not in files or "images_evaluation" not in files:
                logging.warning("no original image files")
                self._download_images()
            # processing: move image to processed path
            self._process_image()

        # check the train-validation-test split information has or hasn't been prepared
        # if not, prepare
        self.path_split = os.path.join(self.path_images, "vinyals_split")
        # check the split information directories do or not exist
        if not os.path.exists(self.path_split):
            logging.warning("no Omniglot split info. path")
            os.makedirs(self.path_split)
        # check the split information files do or not exist,
        # if no files, download
        if len(os.listdir(self.path_split)) < 4:
            logging.warning("no Omniglot split info. files")
            self._download_split_info()

        # define the training or testing tasks root
        # for class attribute print
        super(OmniglotVinyals, self).__init__(self.path_processed,
                                              transform=transform,
                                              target_transform=target_transform)

        # check whether load train, validation, or test task set
        if mode is None:
            logging.info("default loading training task set")
            mode = 'train'
        if mode not in ['train', 'val', 'trainval', 'test']:
            raise ValueError("expected dataset mode should within choices of "
                             "['train', 'val'(validation), 'trainval'(train + validation), 'test'], "
                             "but got {} instead\n".format(mode))

        # prepare data-set
        classes, class_to_idx = self._find_classes_from_vinyals_protocol(mode)
        samples = self._find_items(self.root, class_to_idx, extensions="png")
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported file-type is *.png"))
        # samples_idx_in_classes_idx
        tasks = make_taskset(samples, class_to_idx)

        # setting param
        self.loader = loader
        self.mode = mode
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[-1] for s in samples]
        self.tasks = tasks
        logging.info(self)

    def _download_images(self):
        """
            this function aims to download the Omniglot Dataset,
            and unzip it
            after unzipping, delete the compressed file
        """
        # if no *.zip files, try to download
        files = os.listdir(self.path_images)
        if "images_background.zip" not in files or "images_evaluation.zip" not in files:
            logging.info("starting download Omniglot images")
            for url in self.__img_urls:
                logging.info("--> downloading from %s" % url)
                _data = urllib.request.urlopen(url)
                _file = url.strip().split("/")[-1]
                _file = "\\".join([self.path_images, _file])
                with open(_file, "wb") as f:
                    f.write(_data.read())
            logging.info("successfully download Omniglot images")

        # now try to unzip the compressed *.zip files
        # after unzip, delete the *.zip files
        logging.info("unzip files ...")
        files = ["images_background.zip", "images_evaluation.zip"]
        for _file in files:
            _file = os.path.join(self.path_images, _file)

            # considering the class ZipFile as methods of "__enter__" and "__exit__"
            # so we can use "with ... as ..." for safe operation
            with zipfile.ZipFile(_file, 'r') as _unzip_ref:
                _unzip_ref.extractall(self.path_images)
            os.remove(_file)
        logging.info("successfully unzip Omniglot images")
        return

    def _download_split_info(self):
        """ this function aims to download the Omniglot split information (Vinyals et al. 2016) """
        # if no split info. files, try to download
        files = os.listdir(self.path_split)
        for key, url in self.__vinyals_protocol.items():
            _file = key + '.txt'
            if _file not in files:
                logging.info("starting download %s info. from %s" % (key, url))
                _data = urllib.request.urlopen(url)
                _file = os.path.join(self.path_split, _file)
                with open(_file, "wb") as f:
                    f.write(_data.read())
        logging.info("successfully download Omniglot split info. (Vinyals)")

    def _process_image(self):

        # move images and remove the original files
        for _dir in ['images_background', 'images_evaluation']:
            for _subdir in os.listdir(os.path.join(self.path_images, _dir)):
                shutil.move(os.path.join(self.path_images, _dir, _subdir), self.path_processed)

        # check move successfully
        if len(os.listdir(self.path_processed)) >= 50:
            logging.info("successfully move images into processed dir.")
        else:
            raise FileNotFoundError("moving error")

    def _find_classes_from_vinyals_protocol(self, mode):

        # get current mode classes from vinyals split info. file (with augmentation info)
        cur_mode_file = os.path.join(self.path_split, mode + '.txt')
        cur_class = []
        with open(cur_mode_file, "r") as f:
            for cls in f.readlines():
                cur_class.append(cls.strip())
        cur_class.sort()
        class_to_idx = {cls: i for i, cls in enumerate(cur_class)}
        return cur_class, class_to_idx

    def _find_items(self, dir, class_to_idx, extensions='png'):

        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(class_to_idx.keys()):
            tar_base, _t = target.strip().split(os.altsep + 'rot')
            d = os.path.join(dir, tar_base)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self.__is_valid_file(path, extensions):
                        item = (path, float(_t), class_to_idx[target])
                        images.append(item)
        return images

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
        path, _t, target = self.samples[index]
        sample = self.loader(path).rotate(_t)
        if self.transform is not None:
            sample = self.transform(sample)
            # import numpy as np
            # kkk = sample.numpy()
            # mean, std = np.mean(kkk), np.std(kkk)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

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

    # current version not use this func.
    @staticmethod
    def find_all_class(root):
        """
            this function aims to read the paths of Omniglot images,
            and store the path info
        """
        alphabet_dirs = os.scandir(root)
        classes = []
        for alphabet in alphabet_dirs:
            # the usage of "char.name" is Faster and available in Python 3.5 and above
            classes += [os.path.join(alphabet.name, char.name) for char in os.scandir(alphabet) if char.is_dir()]
        classes.sort()
        return classes


if __name__ == "__main__":

    from torchvision import transforms
    from dataflow.utils import TaskBatchSampler, TaskLoader

    path_cur = os.getcwd()
    path_img = "E:\\Transferring_Datasets\\Omniglot"   # path_cur + "/.."    "E:\\Transferring_Datasets\\Omniglot"
    omi_set = OmniglotVinyals(path_images=path_img,
                              mode="train",
                              transform=transforms.Compose([transforms.Resize((105, 105)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.9], std=[0.3])]))
    omi_task_set = TaskLoader(omi_set, n_way=5, k_shot=1, query_shot=5,
                              batch_shuffle=True,
                              task_shuffle=True,
                              num_workers=2)
    for epoch in range(5):
        for i, (s_samples, s_labels, q_samples, q_labels) in enumerate(omi_task_set):
            print(s_labels, q_labels)
            print(i)
