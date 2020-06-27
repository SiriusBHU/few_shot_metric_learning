import argparse
import os


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # --------
    # data-set
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet',
                                                                                'tieredImageNet',
                                                                                'CIFAR-FS',
                                                                                'FC100'])
    parser.add_argument('--data_root', type=str, default='', help='path to data root')
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')

    # ------------------------------------------
    # pre-training optimization hyper-parameters
    # 1. training epochs
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    # 2. data loader
    parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    # 3. optimizer
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    # 4. lr scheduler
    parser.add_argument('--step_scheduler', action='store_true', help='use step scheduler')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    # -----------------------
    # meta evaluation setting
    parser.add_argument('--eval_iterations', type=int, default=600, metavar='N', help='iterations for evaluation')
    parser.add_argument('--n_way', type=int, default=5, metavar='N', help='Number of classes')
    parser.add_argument('--k_shot', type=int, default=1, metavar='N', help='Number of support shot in each class')
    parser.add_argument('--n_query', type=int, default=15, metavar='N', help='Number of query shot in each class')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='number of augmenting expanded samples for each meta test sample')
    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # --------------------
    # log store parameters
    # saving & print frequency
    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment ID')
    parser.add_argument('--save_log', type=bool, default=True, help='save log or not')
    parser.add_argument('--display_freq', type=int, default=10, help='print evaluation result frequency')
    parser.add_argument('--model_save_freq', type=int, default=10, help='frequency of fixed model saving action')

    # -------
    # bagging
    opt = parser.parse_args()
    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.data_root:
        opt.data_root = os.path.join('./dataset', opt.dataset)
        if opt.dataset == 'CIFAR_FS' or opt.dataset == 'FC100':
            opt.data_root = os.path.join('./dataset', 'CIFAR100')
    else:
        opt.data_root = os.path.join(opt.data_root, opt.dataset)
        if opt.dataset == 'CIFAR_FS' or opt.dataset == 'FC100':
            opt.data_root = os.path.join(opt.data_root, 'CIFAR100')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.transform)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()

    return opt