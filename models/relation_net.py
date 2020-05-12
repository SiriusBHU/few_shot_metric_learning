import torch
import torch.nn as nn
import torch.nn.functional as F
from models.metric_loss import get_similarity_matrix, get_label_matrix, relation_loss


class ConvBlock(nn.Module):

    def __init__(self, in_chs, out_chs):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 3, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class RelationNet(nn.Module):

    r"""
        Arguments:
            in_chs (int): input channels
            hidden_chs (int): hidden layer channels
            out_chs (int): output channels
    """

    def __init__(self,
                 in_chs, hidden_chs, out_chs,
                 in_dims=84):
        super(RelationNet, self).__init__()

        self.extractor = nn.Sequential(
            ConvBlock(in_chs, hidden_chs),
            nn.MaxPool2d(2),
            ConvBlock(hidden_chs, hidden_chs),
            nn.MaxPool2d(2),
            ConvBlock(hidden_chs, hidden_chs),
            ConvBlock(hidden_chs, out_chs),
        )

        self.conv_comparator = nn.Sequential(
            ConvBlock(2 * out_chs, hidden_chs),
            nn.MaxPool2d(2),
            ConvBlock(hidden_chs, hidden_chs),
            nn.MaxPool2d(2),
        )

        self.fc_comparator = nn.Sequential(
            nn.Linear(in_dims // 16, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def embedding(self, x):
        """ extracting features for distance computation """
        x = self.extractor(x)
        return x.view(x.size(0), -1)

    def relation_score(self, x1, x2):
        nums_q, nums_s, chs, h, w = x1.size(0), x1.size(1), x1.size(2), x1.size(3), x1.size(4)
        x = torch.cat([x1, x2], dim=2).reshape(nums_q * nums_s, 2 * chs, h, w)
        x = self.conv_comparator(x)
        x = self.fc_comparator(x.view(x.size(0), -1))
        return x.view(nums_q, nums_s)

    @staticmethod
    def relation_gather(relation_matrix, label_matrix, label_s):

        label = torch.unique(label_s)
        relation_matrix = torch.cat([torch.sum(relation_matrix[:, label_s == c], dim=1).view(-1, 1)
                                     for c in label], dim=1)
        label_matrix = torch.cat([torch.sum(label_matrix[:, label_s == c], dim=1).view(-1, 1)
                                     for c in label], dim=1)
        return relation_matrix, label_matrix, label

    def forward(self, img_s, img_q, label_s, label_q):

        # compute embedding
        embeddings_s, embeddings_q = self.embedding(img_s), self.embedding(img_q)

        # compute similarity and label matrix
        relation_matrix = get_similarity_matrix(embeddings_s, embeddings_q, self.relation_score)
        label_matrix = get_label_matrix(label_s, label_q)
        relation_matrix, label_matrix, label = self.relation_gather(relation_matrix, label_matrix, label_s)

        # compute the loss
        loss = relation_loss(relation_matrix, label_matrix)
        # compute the accuracy
        predict_q = label[torch.argmin(relation_matrix, dim=1)]
        correct_q = predict_q == label_q
        acc = torch.sum(correct_q).item() / label_q.size(0)

        return acc, loss








