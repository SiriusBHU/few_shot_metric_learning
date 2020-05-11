import torch
import torch.nn as nn
from models.metric_loss import get_similarity_matrix, get_label_matrix, pairwise_loss


class SiameseNet(nn.Module):

    r"""
        Arguments:
            is_balance (bool, optional): whether to keep the the number of positive
            sample pairs is almost equal to that of negative samples.
            (default: False)
    """

    def __init__(self, is_balance=False):
        super(SiameseNet, self).__init__()
        pass

        self.is_balance = is_balance
        self.conv_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10, stride=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7, stride=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=4, stride=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, bias=True),
            nn.ReLU()
        )
        self.fc_extractor = nn.Sequential(
            nn.Linear(in_features=6*6*256, out_features=4096, bias=True),
            nn.Sigmoid()
        )
        self.metric = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def embedding(self, x):
        """ extracting features for L1 distance computation """
        x = self.conv_extractor(x)
        x = self.fc_extractor(x.view(x.size(0), -1))
        return x

    def dis_func(self, _support_expanded, embeddings_que):
        num_q, num_s = _support_expanded.size(0), _support_expanded.size(1)
        _dis = torch.abs(_support_expanded - embeddings_que).view(num_s * num_q, -1)
        return self.metric(_dis)

    def forward(self, img_s, img_q, label_s, label_q):

        # compute embedding
        embeddings_s, embeddings_q = self.embedding(img_s), self.embedding(img_q)

        # compute similarity and label matrix
        similarity_matrix = get_similarity_matrix(embeddings_s, embeddings_q, self.dis_func)
        label_matrix = get_label_matrix(label_s, label_q)

        # compute the loss
        loss = pairwise_loss(similarity_matrix, label_matrix)

        # compute the accuracy
        predict_q = label_s[torch.argmax(similarity_matrix, dim=1)]
        correct_q = predict_q == label_q
        acc = torch.sum(correct_q).item() / label_q.size(0)

        return acc, loss








