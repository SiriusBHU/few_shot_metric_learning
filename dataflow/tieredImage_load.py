"""
    Tiered-ImageNet Data-Set ReadFile
    Author: Sirius HU
    Created Date: 2020.05.15

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
import cv2
import shutil
from torchvision.datasets.vision import VisionDataset
from dataflow.utils import make_taskset, pil_rgb_loader


class TieredImageNet(VisionDataset):

    r"""
        Arguments:
            path_images (string, optional): the directory store TieredImageNet data.
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
    __img_urls = "https://drive.google.com/uc?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07&export=download"
    __ren_protocol = ('test', 'train', 'val')

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
            self.path_images = os.path.join(os.getcwd(), "dataset\\tieredImageNet")

        # ----------------
        # check path exist
        # check tiered-ImageNet main path
        if not os.path.exists(self.path_images):
            logging.warning("no tiered-ImageNet data-set path")
            os.makedirs(self.path_images)
        # check the processed path
        self.path_processed = os.path.join(self.path_images, 'images')
        if not os.path.exists(self.path_processed):
            logging.warning("no tiered-ImageNet processed images' path")
            os.makedirs(self.path_processed)
        # check the split information path
        self.path_split = os.path.join(self.path_images, "split")
        if not os.path.exists(self.path_split):
            logging.warning("no tiered-ImageNet split info. path")
            os.makedirs(self.path_split)
        # check the split information directories do or not exist
        self.path_pickle = os.path.join(self.path_images, 'pickle_files')
        if not os.path.exists(self.path_pickle):
            logging.warning("no mini-ImageNet pickle path")
            os.makedirs(self.path_pickle)

        # ------------------------------------------
        # check the images and the corresponding
        # split info have or not been prepared,
        # if not, prepare them
        if not len(os.listdir(self.path_processed)) \
                or len(os.listdir(self.path_split)) < 3:
            logging.info("the tiered image has not been processed as image files")

            # if no processed images or split info.,
            # check if there have the corresponding pickle files
            if not os.path.exists(self.path_pickle) \
                    or len(os.listdir(self.path_pickle)) < 8:
                raise FileExistsError("pickled files of tiered-ImageNet should be download first\n"
                                      "then please unzip the pickle files into '%s' .\n"
                                      "reference download url: %s"
                                      % (self.path_pickle, self.__img_urls))
            self._unpickle()

        # --------------------------------------
        # after guaranteeing the images has been
        # prepared in the setting path_images,
        # set root and declare its father class
        self.root = self.path_processed
        super(TieredImageNet, self).__init__(self.root,
                                             transform=transform,
                                             target_transform=target_transform)

        # ---------------
        # prepare dataset

        # check whether load train, validation, or test task set
        if mode is None:
            logging.info("default loading training task set")
            mode = 'train'
        if mode not in ['train', 'val', 'test']:
            raise ValueError("expected data-set mode should within choices of "
                             "['train', 'val'(validation), 'test'], "
                             "but got {} instead\n".format(mode))

        # generate dataset
        classes, class_to_idx, samples = self._find_classes_items(mode, 'jpg')
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
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

    def _unpickle(self):
        """
            unpickle files,
            store images into processed directory, and
            save split information into split directory
        """
        for mode in self.__ren_protocol:
            d_file, l_file = mode + '_images_png.pkl', mode + '_labels.pkl'
            with open(os.path.join(self.path_pickle, d_file), 'rb') as f:
                data = pickle.load(f)
                logging.info("successfully unzip the [%s] pickled image-files" % mode)
                for i, _d in enumerate(data):
                    _d = cv2.imdecode(_d, 1)
                    _f = os.path.join(self.path_processed, mode + str(i).zfill(7) + '.jpg')
                    cv2.imwrite(filename=_f, img=_d)

            with open(os.path.join(self.path_pickle, l_file), 'rb') as f:
                labels = pickle.load(f)
                labels["idx_to_file_names"] = [mode + str(idx).zfill(7) + '.jpg'
                                               for idx in range(len(labels['label_specific']))]
                with open(os.path.join(self.path_split, mode), 'wb') as f:
                    pickle.dump(labels, f)

            # release memory
            import gc
            del data, labels
            gc.collect()

        logging.info("images have been processed into directory: %s\n"
                     "split information has been stored into directory: %s "
                     % (self.path_processed, self.path_split))

    def _find_classes_items(self, mode, extensions='jpg'):

        # get ren split info. file
        if mode != 'trainval':
            cur_mode_file = os.path.join(self.path_split, mode)
            with open(cur_mode_file, 'rb') as f:
                label_info = pickle.load(f)
        else:
            with open(os.path.join(self.path_split, 'train'), 'rb') as f:
                label_info = pickle.load(f)

            with open(os.path.join(self.path_split, 'test'), 'rb') as f:
                _info = pickle.load(f)
                _info['label_specific'] += len(label_info['label_specific_str'])
                _info['label_general'] += len(label_info['label_general_str'].keys())

                import numpy as np
                label_info['label_specific'] = np.concatenate([_info['label_specific'],
                                                               label_info['label_specific']],
                                                              axis=0)
                label_info['label_general'] = np.concatenate([_info['label_general'],
                                                              label_info['label_general']],
                                                             axis=0)
                label_info['label_specific_str'].extend(_info['label_specific_str'])

        # get classes and the corresponding index of the current mode
        classes = label_info['label_specific_str']
        class_to_idx = {_cls: i for i, _cls in enumerate(classes)}

        # formulate samples-set and the corresponding tasks-set
        images = []

        import time
        t1 = time.time()
        for _s, _cls in zip(label_info['idx_to_file_names'], label_info['label_specific'],):
            _file = os.path.join(self.root, _s)
            if self.__is_valid_file(_file, extensions):
                if os.path.exists(_file):
                    item = (_file, _cls)
                    images.append(item)
                else:
                    continue
        print(time.time() - t1)
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
    from dataflow.utils import TaskBatchSampler, TaskLoader
    import time

    path_cur = os.getcwd()
    path_img = "E:\\Transferring_Datasets\\Tiered_ImageNet"   # path_cur + "/.."    "E:\\Transferring_Datasets\\Omniglot"
    tiered_img_set = TieredImageNet(path_images=path_img,
                                    mode="train",
                                    transform=transforms.Compose([transforms.Resize((84, 84)),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                                   std=(0.229, 0.224, 0.225))]))
    tiered_img_loader = TaskLoader(tiered_img_set,
                                   iterations=100,
                                   n_way=5, k_shot=1, query_shot=5,
                                   batch_shuffle=True,
                                   task_shuffle=True,
                                   num_workers=2)
    t1 = time.time()
    for i, (s_samples, s_labels, q_samples, q_labels) in enumerate(tiered_img_loader):
        t2 = time.time()
        print(i, ": ", t2 - t1)
        t1 = t2
