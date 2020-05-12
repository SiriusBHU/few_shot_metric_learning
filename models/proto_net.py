import torch
import torch.nn as nn
import torch.nn.functional as F
from models.metric_loss import get_similarity_matrix, get_label_matrix, proto_loss, euclidean_dis


class ProtoBlock(nn.Module):

    def __init__(self, in_chs, out_chs):
        super(ProtoBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 3, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class ProtoNet(nn.Module):

    r"""
        Arguments:
            in_chs (int): input channels
            hidden_chs (int): hidden layer channels
            out_chs (int): output channels
    """

    def __init__(self,
                 in_chs, hidden_chs, out_chs,
                 metric_type='Euclidean'):
        super(ProtoNet, self).__init__()

        self.extractor = nn.Sequential(
            ProtoBlock(in_chs, hidden_chs),
            ProtoBlock(hidden_chs, hidden_chs),
            ProtoBlock(hidden_chs, hidden_chs),
            ProtoBlock(hidden_chs, out_chs),
        )
        if not isinstance(metric_type, str) or metric_type not in ["Euclidean", "Cosine"]:
            raise AttributeError("expected metric type is within ['Euclidean', 'Cosine'], "
                                 "but got {}".format(metric_type))
        self.metric_type = metric_type

    def embedding(self, x):
        """ extracting features for distance computation """
        x = self.extractor(x)
        return x.view(x.size(0), -1)

    @staticmethod
    def proto_compute(embeddings_s, label_s):
        """ compute the prototype of each class """
        label_proto = torch.unique(label_s)

        _idxs = [label_s == c for c in label_proto]

        feature_proto = torch.stack([torch.mean(embeddings_s[label_s == c], dim=0) for c in label_proto])
        return feature_proto, label_proto

    def forward(self, img_s, img_q, label_s, label_q):

        # compute embedding
        embeddings_s, embeddings_q = self.embedding(img_s), self.embedding(img_q)
        embeddings_proto, label_proto = self.proto_compute(embeddings_s, label_s)

        # compute distance and label matrix
        dist_matrix = get_similarity_matrix(embeddings_proto, embeddings_q, euclidean_dis)
        label_matrix = get_label_matrix(label_proto, label_q)

        # compute the loss
        loss = proto_loss(dist_matrix, label_matrix)    # similarity_matrix
        # compute the accuracy
        predict_q = label_proto[torch.argmin(dist_matrix, dim=1)]
        correct_q = predict_q == label_q
        acc = torch.sum(correct_q).item() / label_q.size(0)

        return acc, loss








