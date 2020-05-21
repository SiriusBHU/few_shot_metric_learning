import torch
import torch.optim as optim
from torch.nn.init import kaiming_uniform_
from dataflow.omniglot_load import OmniglotVinyals
from dataflow.miniImage_load import MiniImageNet
from dataflow.utils import TaskLoader
from torchvision import transforms
import time
import os


def data_loader_preparation(data_choice,
                            transform_opt,
                            train_iterations, test_iterations,
                            train_way, train_shot, train_query,
                            test_way, test_shot, test_query,
                            batch_shuffle, task_shuffle):

    if data_choice == 'Omniglot':
        path_img = "E:\\Transferring_Datasets\\Omniglot"
        SetInit = OmniglotVinyals
    elif data_choice == 'miniImageNet':
        path_img = "D:\\Transferring_Datasets\\Mini_ImageNet"
        SetInit = MiniImageNet
    else:
        raise AttributeError("error")

    def init_dataset(path_images, mode):
        return SetInit(path_images, mode=mode, transform=transform_opt)

    train_set = init_dataset(path_img, "train")
    val_set = init_dataset(path_img, "val")
    test_set = init_dataset(path_img, "test")

    def init_loader(dataset, iterations, way, shot, query):
        return TaskLoader(dataset,
                          iterations=iterations,
                          n_way=way, k_shot=shot, query_shot=query,
                          batch_shuffle=batch_shuffle,
                          task_shuffle=task_shuffle,
                          num_workers=2)

    train_loader = init_loader(train_set, train_iterations, train_way, train_shot, train_query)
    val_loader = init_loader(val_set, test_iterations, test_way, train_shot, train_query)
    test_loader = init_loader(test_set, test_iterations, test_way, test_shot, test_query)

    return train_loader, val_loader, test_loader


def weight_init(m):

    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        kaiming_uniform_(m.weight.data)
    if class_name.find('Linear') != -1:
        kaiming_uniform_(m.weight.data)
    if class_name.find('BatchNorm') != -1:
        torch.nn.init.uniform_(m.weight)


class NetFlow(object):

    def __init__(self,
                 net,
                 optimizer,
                 lr_scheduler,
                 epochs,
                 iterations,
                 display_epoch=5):

        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # epoch to display evaluation result
        self.epochs = epochs
        self.display_epoch = display_epoch
        self.iterations = iterations

        # set path for param dump and load
        self.path_cur = os.getcwd()

    def train(self, train_loader, val_loader, net_name, is_dump=False):

        # dump path generate
        if is_dump:
            self.make_dump_directory(net_name)

        # set net, optimizer, lr_scheduler
        net = self.net
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        # initialization
        net.apply(weight_init)
        net.cuda()

        # training period
        LOSS_train, ACC_train, LOSS_val, ACC_val = [], [], [], []
        MAX_ACC = 0
        for epoch in range(EPOCHS):

            net.train()
            acc_train, loss_train, t1 = [], [], time.time()
            for i, (imgs_s, labels_s, imgs_q, labels_q) in enumerate(train_loader):
                # feed-forward
                imgs_s, imgs_q = imgs_s.cuda(), imgs_q.cuda()
                labels_s, labels_q = labels_s.cuda(), labels_q.cuda()
                _acc, _loss = net(imgs_s, imgs_q, labels_s, labels_q)

                # feed-back
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

                acc_train.append(_acc), loss_train.append(_loss.item())
                print("[iteration: %3d] -- [loss: %.5f] -- [acc: %.5f]" %
                      (i + 1, _loss.item(), _acc))
            # record result and show
            LOSS_train.append(loss_train[:self.iterations])
            ACC_train.append(acc_train[:self.iterations])

            # upgrade learning rate
            lr_scheduler.step(epoch)

            # ------- evaluate the network on validation_loader --------
            net.eval()
            acc_val, loss_val = [], []
            with torch.no_grad():
                for i, (imgs_s, labels_s, imgs_q, labels_q) in enumerate(val_loader):
                    imgs_s, imgs_q = imgs_s.cuda(), imgs_q.cuda()
                    labels_s, labels_q = labels_s.cuda(), labels_q.cuda()
                    _acc, _loss = net(imgs_s, imgs_q, labels_s, labels_q)

                    acc_val.append(_acc), loss_val.append(_loss.item())
            LOSS_val.append(loss_val[:self.iterations])
            ACC_val.append(acc_val[:self.iterations])
            print("Epoch: %3d -- Time Consumption: %.5fs\n"
                  "====> Training:  -- [loss: %.5f] -- [acc: %.5f]\n"
                  "====> Validation:-- [loss: %.5f] -- [acc: %.5f]\n" %
                  (epoch + 1, time.time() - t1,
                   sum(loss_train) / len(loss_train), sum(acc_train) / len(acc_train),
                   sum(loss_val) / len(loss_val), sum(acc_val) / len(acc_val)))
            if (epoch + 1) % self.display_epoch == 0 and is_dump:
                self.dump_param(net_name, epoch + 1)
            if sum(acc_val) / len(acc_val) > MAX_ACC:
                self.dump_param(net_name + '_Best_ACC', epoch + 1)
                MAX_ACC = sum(acc_val) / len(acc_val)

        return LOSS_train, ACC_train, LOSS_val, ACC_val

    def evaluation(self, test_loader):
        self.net.eval()
        acc_test, loss_test = [], []
        with torch.no_grad():
            for i, (imgs_s, labels_s, imgs_q, labels_q) in enumerate(test_loader):
                imgs_s, imgs_q = imgs_s.cuda(), imgs_q.cuda()
                labels_s, labels_q = labels_s.cuda(), labels_q.cuda()
                _acc, _loss = self.net(imgs_s, imgs_q, labels_s, labels_q)
                loss_test.append(_loss.item()), acc_test.append(_acc)
        print("====> Testing -- [loss: %.5f] -- [acc: %.5f]" %
              (sum(loss_test) / len(loss_test), sum(acc_test) / len(acc_test)))

    def make_dump_directory(self, net_name):

        # set parameter dump directory
        cur_time = time.strftime('%Y.%m.%d_%H.%M.%S', time.localtime(time.time()))
        self.path_dump = self.path_cur + "\\dataflow\\paramcache\\%s_%s" % (net_name, cur_time)

        if not os.path.exists(self.path_dump):
            os.mkdir(self.path_dump)
        return

    def load_param(self, net_name, param_file=None, param_path=None):

        if not param_path:
            path_dump = os.path.join(self.path_cur, "dataflow\\paramcache")
            dump_dirs = []
            for item in os.listdir(path_dump):
                if net_name in item:
                    dump_dirs.append(item)
            param_path = os.path.join(path_dump, dump_dirs[-1])

        if not param_file:
            param_file = os.listdir(param_path)[-1]

        load_file = os.path.join(param_path, param_file)
        self.net.cpu()
        self.net.load_state_dict(torch.load(load_file))
        self.net.cuda()

    def dump_param(self, net_name, step):

        dump_name = os.path.join(self.path_dump, '%s_params_%s.pkl' % (net_name, str(step).zfill(4)))
        torch.save(self.net.state_dict(), dump_name)
        return


if __name__ == "__main__":

    # clear CUDA memory
    torch.cuda.empty_cache()

    # parameter setting
    DATA_CHOICE = 'miniImageNet'    # 'miniImageNet', 'Omniglot'
    # TRANSFORM_OPT = transforms.Compose([transforms.Resize((84, 84)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[0.9], std=[0.3])])
    TRANSFORM_OPT = transforms.Compose([transforms.Resize((84, 84)),
                                        transforms.CenterCrop(84),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))])
    TRAIN_WAY, TRAIN_SHOT, TRAIN_QUERY = 30, 1, 15
    TEST_WAY, TEST_SHOT, TEST_QUERY = 5, 1, 30
    BATCH_SHUFFLE, TASK_SHUFFLE = True, True
    EPOCHS = 200
    TRAIN_ITERATIONS = 100
    TEST_ITERATIONS = 100
    DISPLAY = 10

    # net setting
    from models.siamese_net import SiameseNet
    from models.proto_net import ProtoNet
    NET, NET_NAME = ProtoNet(3, 64, 64), "ProtoNet_Image"  # "ProtoNet_Image", "ProtoNet_Omniglot"
    OPTIMIZER = optim.Adam(NET.parameters(), lr=1e-3)
    LR_SCHE = optim.lr_scheduler.StepLR(OPTIMIZER, step_size=20, gamma=0.5)


    # data_preparation
    train_loader, val_loader, test_loader = data_loader_preparation(DATA_CHOICE,
                                                                    TRANSFORM_OPT,
                                                                    TRAIN_ITERATIONS, TEST_ITERATIONS,
                                                                    TRAIN_WAY, TRAIN_SHOT, TRAIN_QUERY,
                                                                    TEST_WAY, TEST_SHOT, TEST_QUERY,
                                                                    BATCH_SHUFFLE, TASK_SHUFFLE)
    # net flow instance

    net_flow = NetFlow(NET, optimizer=OPTIMIZER, lr_scheduler=LR_SCHE,
                       epochs=EPOCHS,
                       iterations=TRAIN_ITERATIONS,
                       display_epoch=DISPLAY)
    # training
    LOSS_train, ACC_train, LOSS_val, ACC_val = net_flow.train(train_loader, val_loader,
                                                              NET_NAME,
                                                              is_dump=True)
    # evaluation on testing set
    net_flow.evaluation(test_loader)

    # load parameter and evaluation on testing set
    net_flow.load_param(NET_NAME)
    net_flow.evaluation(test_loader)

    # show train result
    import numpy as np
    LOSS, ACC = np.array([LOSS_train, LOSS_val]), np.array([ACC_train, ACC_val])
    LOSS, ACC = LOSS.reshape(2, -1), ACC.reshape(2, -1)
    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.plot(LOSS[0], lw=0.7, c='k')
    plt.plot(LOSS[1], lw=0.7, c='r')
    plt.subplot(212)
    plt.plot(ACC[0], lw=0.7, c='k')
    plt.plot(ACC[1], lw=0.7, c='r')
    plt.show()











