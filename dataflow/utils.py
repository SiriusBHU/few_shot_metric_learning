from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader


def pil_rgb_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def pil_grey_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def pil_array_to_image(arr):
    return Image.fromarray(arr)


def make_taskset(samples, class_to_idx):
    """
        find the images' indexes of each classes
        output a dict as {class index: [corresponding image indexes]}
    """
    # initial task set
    tasks = dict()
    for _class, _class_idx in class_to_idx.items():
        tasks[_class_idx] = []

    # add the sample index to the list of the sample's task set
    for _idx, _s in enumerate(samples):
        tasks[_s[-1]].append(_idx)

    return tasks


class MetaBatchSampler(object):

    r"""
        Arguments:
            data_source (base-class Dataset): dataset from  which to load the data.
            iterations (None, int, optional): how many iterations per epoch to load
                (default: None)
            n_way (int, optional): how many classes per task to load
                (default: 2).
            k_shot (int, optional): how many samples per task to load for support set
                (default: 2).
            query_shot (int, None, optional): how many samples per task to load for query set
                (default: 2).
            task_shuffle (bool, optional): set to 'True' to have the classes reshuffled at
                every epoch (default: False).
            drop_last (bool, optional): set to ``True`` to drop the last unformulated task,
                if the classes' number of the data-set is not divisible by the 'n_way'
                (classes number per task).
                If ``False`` and the classes' number  is not divisible by the task batch size,
                then the last task batch will be smaller (default: ``True``).
    """

    def __init__(self,
                 data_source,
                 iterations=None,
                 n_way=2, k_shot=1, query_shot=None,
                 task_shuffle=False,
                 batch_shuffle=False,
                 drop_last=True):

        self.data_source = data_source

        # check attribute
        if iterations is not None:
            if not isinstance(iterations, (int, np.int)) or n_way < 1:
                raise ValueError(("param iterations must be 'None', 'int or np.int >= 1', "
                                  "but got iterations={}").format(iterations))
        if not isinstance(n_way, (int, np.int)) or n_way < 2:
            raise ValueError(("param n_way must be 'int or np.int' >= 2, "
                              "but got n_way={}").format(n_way))
        if not isinstance(k_shot, (int, np.int)) or k_shot < 1:
            raise ValueError(("param k_shot must be 'int or np.int' >= 1, "
                              "but got k_shot={}").format(k_shot))
        if query_shot is not None:
            if not isinstance(query_shot, (int, np.int)) or query_shot < 1:
                raise ValueError(("param query_shot must be 'int or np.int' >= 1, "
                                  "but got query_shot={}").format(query_shot))

        if not isinstance(task_shuffle, bool):
            raise ValueError(("param task_shuffle must be 'bool' type, "
                              "but got task_shuffle={}").format(task_shuffle))
        if not isinstance(batch_shuffle, bool):
            raise ValueError(("param batch_shuffle must be 'bool' type, "
                              "but got batch_shuffle={}").format(batch_shuffle))

        # set iterations
        self.iterations = iterations
        # n-way k-shot query-shot setting
        self.n_way = n_way
        self.total_shot = k_shot + query_shot if query_shot is not None else None
        self.k_shot = k_shot
        self.query_shot = query_shot
        # if index loading shuffle
        self.task_shuffle = task_shuffle
        self.batch_shuffle = batch_shuffle

        self.drop_last = drop_last
        self.cur_task_num = [0]
        self._num_samples = self.get_samples_num()
        self._num_tasks = self.get_tasks_num()

    def get_samples_num(self):
        # data-set size might change at runtime
        return len(self.data_source)

    def get_tasks_num(self):
        return len(self.data_source.tasks.keys())

    def __iter__(self):

        if self.iterations is None:
            self.iterations = self._num_tasks // self.n_way
        if not self.drop_last:
            self.iterations += 1

        # prepare task perm
        classes_perm = torch.arange(self._num_tasks)
        if self.task_shuffle:
            classes_perm = torch.randperm(self._num_tasks)

        _start_idx, _end_idx = 0, self.n_way
        for _ in range(self.iterations):

            # 双线程读取数据时，需要在这里更新 cur_task_num， 否则容易出现错误
            self.cur_task_num, task_batch = [0], []
            for idx in classes_perm[_start_idx: _end_idx].tolist():
                # prepare batch
                _cur_len = len(self.data_source.tasks[idx])
                _cur_shot = self.total_shot if self.total_shot is not None else _cur_len
                if self.batch_shuffle:
                    samples_perm = torch.randperm(_cur_len).tolist()
                    _batch = [self.data_source.tasks[idx][samples_perm[i]] for i in range(_cur_shot)]
                else:
                    _batch = self.data_source.tasks[idx][:_cur_shot]
                task_batch += _batch

                # use the cur_task_idx to record the tasks o f the current batch
                # this is used for support-query sets split
                self.cur_task_num.append(len(task_batch))
            yield task_batch

            # update task idx
            _start_idx, _end_idx = _start_idx + self.n_way, _end_idx + self.n_way
            # check if need update perm and re-start
            if _start_idx >= self._num_tasks or \
                (_end_idx > self._num_tasks and self.drop_last):
                if self.task_shuffle:
                    classes_perm = torch.randperm(self._num_tasks)
                _start_idx, _end_idx = 0, self.n_way
            # if drop-last, check if it is the end
            elif _end_idx > self._num_tasks and not self.drop_last:
                _end_idx = self._num_tasks

    def __len__(self):
        return self._num_tasks


class MetaTaskLoader(object):
    r"""
        Arguments:
            data_source (base-class Dataset): dataset from  which to load the data.
            n_way (int): how many classes per task to load
                (default: 2).
            k_shot (int): how many samples per class to load for each episode as support set
                (default:  1).
            query_shot (int, None): how many samples per class to load for each episode as query set.
                default setting 'None' means using all the residual samples as query set.
                (default:  None).
            task_shuffle (bool, optional): set to 'True' to have the classes reshuffled at
                every epoch (default: False).
            batch_shuffle (bool, optional): set to 'True' to have the  k-shot batch in each
                class reshuffled at every epoch (default: False).
            drop_last (bool): set to ``True`` to drop the last unformulated task, if the classes'
                number of the data-set is not divisible by the 'n_way' (classes number per task).
                If ``False`` and the classes' number  is not divisible by the task batch size,
                then the last task batch will be smaller (default: ``True``).
            task_batch_sampler (Sampler, optional): each time returns a batch of indices, which
                contains the samples' indexes of the current n-way k-shot task (n classes, each
                class has k samples). Mutually exclusive with :attr:`task_shuffle`,
                :attr:`drop_last`. (default: None)
    """
    def __init__(self, data_source,
                 iterations=None,
                 n_way=2, k_shot=1, query_shot=None,
                 num_workers=0,
                 batch_shuffle=False,
                 task_shuffle=False,
                 drop_last=True,
                 task_batch_sampler=None):

        self.data_source = data_source

        # check attr. n-way, k-shot and batch_shuffle
        if iterations is not None:
            if not isinstance(iterations, (int, np.int)) or n_way < 1:
                raise ValueError(("param iterations must be 'None', 'int or np.int >= 1', "
                                  "but got iterations={}").format(iterations))
        if not isinstance(n_way, (int, np.int)) or n_way < 2:
            raise ValueError(("param n_way must be 'int' or 'np.int' >= 2, "
                              "but got n_way={}").format(n_way))
        if not isinstance(k_shot, (int, np.int)) or k_shot < 1:
            raise ValueError(("param k_shot must be 'int or np.int' >= 1, "
                              "but got k_shot={}").format(k_shot))
        if query_shot is not None:
            if not isinstance(query_shot, (int, np.int)) or query_shot < 1:
                raise ValueError(("param query_shot must be 'int or np.int' >= 1, "
                                  "but got k_shot={}").format(query_shot))

        # check if the task_batch_sampler exists
        #   if has task_batch_sampler:
        #       check its exclusive attr. 'task_shuffle', 'drop_last' and 'n_way', and
        #       check whether its data_source is the same as the current task loader's data_source
        #   if not: accordingly create a task_batch_sampler
        if task_batch_sampler is not None:
            # check attr exclusive
            if task_shuffle or batch_shuffle or not drop_last or iterations is not None or \
                    n_way != 2 or k_shot != 1 or query_shot is not None:
                raise ValueError('task_batch_sampler option is mutually exclusive '
                                 'with task_shuffle, batch_shuffle, iterations, '
                                 'n_way, k_shot, query_shot and drop_last')
            if self.data_source.root != task_batch_sampler.data_source.root:
                raise ValueError('the data source of task_batch_sampler must be '
                                 'the same as that of task loader (self), but got\n'
                                 'sampler-source:   %s\n'
                                 'loader-source:    %s\n'
                                 % (self.data_source.root, task_batch_sampler.data_source.root))
            task_shuffle, drop_last = None, None
            t = task_batch_sampler
            iterations, n_way, k_shot, query_shot = t.iterations, t.n_way, t.k_shot, t.query_shot
        else:
            task_batch_sampler = MetaBatchSampler(self.data_source,
                                                  iterations=iterations,
                                                  task_shuffle=task_shuffle,
                                                  batch_shuffle=batch_shuffle,
                                                  drop_last=drop_last,
                                                  n_way=n_way, k_shot=k_shot, query_shot=query_shot)

        # check if n-way, k-shot, query-shot setting is satisfied
        self._check_n_way_k_shot_satisfied(n_way, k_shot, query_shot)

        # set param
        self.iterations=iterations
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shot = query_shot
        self.task_batch_sampler = task_batch_sampler
        self.data_loader = DataLoader(self.data_source,
                                      batch_sampler=self.task_batch_sampler,
                                      num_workers=num_workers, pin_memory=True)
        self.batch_shuffle = batch_shuffle

    def _check_n_way_k_shot_satisfied(self, n_way, k_shot, query_shot):
        """ check the current loader's data-source satisfaction of n_way k_shot setting"""
        if len(self.data_source.tasks) < n_way:
            raise AttributeError("expected at least %d-way classes, but got %d instead"
                                 % (n_way, len(self.data_source.tasks)))

        for task, idx_in_task in self.data_source.tasks.items():
            if query_shot is None:
                query_shot = 1
            if len(idx_in_task) < k_shot + query_shot:
                raise AttributeError("expected at least %d-shot samples, "
                                     "but in '%s' got only %d samples instead"
                                     % (k_shot + query_shot,
                                        self.data_source.classes[task],
                                        len(idx_in_task)))

    def __iter__(self):

        for samples, labels in self.data_loader:
            cur_task_starts = self.task_batch_sampler.cur_task_num
            s_samples, q_samples, s_labels, q_labels = [], [], [], []

            for sta_idx, end_idx in zip(cur_task_starts[:-1], cur_task_starts[1:]):
                s_samples.append(samples[sta_idx: sta_idx + self.k_shot])
                s_labels.append(labels[sta_idx: sta_idx + self.k_shot])
                q_samples.append(samples[sta_idx + self.k_shot: end_idx])
                q_labels.append(labels[sta_idx + self.k_shot: end_idx])
            s_samples = torch.cat(s_samples, dim=0)
            q_samples = torch.cat(q_samples, dim=0)
            s_labels = torch.cat(s_labels, dim=0)
            q_labels = torch.cat(q_labels, dim=0)
            yield s_samples, s_labels, q_samples, q_labels

    def __repr__(self):
        body = [self.__class__.__name__ + " of " + self.data_source.__class__.__name__,
                "\tTask Formulation:",
                "\t\t%d-way %d-shot" % (self.n_way, self.k_shot),
                "\t\tquery shot setting as {}".format(self.query_shot if self.query_shot is not None
                                                      else "all the residual imgs each class"),
                "\tIs_batch_shuffle: '{}'\n".format(self.batch_shuffle)]
        return '\n'.join(body)