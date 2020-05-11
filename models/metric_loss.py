import torch
import torch.nn.functional as F


def euclidean_dis(_s_expanded, _q_expanded):
    return torch.sum(torch.pow(_s_expanded - _q_expanded, 2), dim=-1)


def l1_dis(_s_expanded, _q_expanded):
    return torch.sum(torch.abs(_s_expanded - _q_expanded), dim=-1)


def pairwise_expand(_s, _q):
    _s = _s.view(-1, 1) if len(_s.size()) == 1 else _s
    _q = _q.view(-1, 1) if len(_q.size()) == 1 else _q

    num_q, num_s = _q.size(0), _s.size(0)
    dims = _s.size(1)
    if dims != _q.size(1):
        raise AttributeError("support set features do not match query set features, "
                             "expected support_dim=%d, but got query_dim = %d!\n"
                             % (dims, _q.size(1)))
    _q = _q.unsqueeze(1).expand(num_q, num_s, dims)
    _s = _s.unsqueeze(0).expand(num_q, num_s, dims)

    return _q, _s


def get_similarity_matrix(embeddings_support, embeddings_query, dis_func=euclidean_dis):
    r"""
        compute the similarity matrix of each samples from support-set and query-set

        Arguments:
            embeddings_support (Tensor): the tensor of support examples
            embeddings_query (Tensor): the tensor of query examples
            dis_func (callable): the distance function, which metric the similarity
                between each samples from support-set and query-set
                (default: euclidean_dis)
        Returns:
            (Tensor): A tensor with two dimensions containing the similarity info.
        Example::
            input: x_support -- x_support.size() = (18 [nums], 32 [feature dims])
                   x_query -- x_query.size() = (54 [nums], 32 [feature dims])
                   dis_func
            output: similarity_matrix -- similarity_matrix.size = (54, 18)
    """

    # expand the support & query samples
    _s_expanded, _q_expanded = pairwise_expand(embeddings_support, embeddings_query)

    # re-size for distance compute
    num_q, num_s = _s_expanded.size(0), _s_expanded.size(1)
    _similarity = dis_func(_s_expanded, _q_expanded)
    return _similarity.view(num_q, num_s)


def get_label_matrix(labels_support, labels_query):

    r"""
        compute the similarity matrix of each samples from support-set and query-set

        Arguments:
            labels_support (Tensor): the tensor of support examples' labels
            labels_query (Tensor): the tensor of query examples' labels
        Returns:
            (Tensor): A tensor with two dimensions containing the similarity info.
        Example::
            input: labels_support = tensor([1, 1, 2, 3])
                   labels_query= tensor([1, 2, 3])
            output: tensor([[1, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    """
    # expand the support & query labels
    labels_support, labels_query = pairwise_expand(labels_support, labels_query)

    # compare to get label matrix
    _matrix = labels_support == labels_query
    _matrix = _matrix.to(dtype=torch.long)

    num_q, num_s = _matrix.size(0), _matrix.size(1)
    return _matrix.view(num_q, num_s)


def pairwise_loss(sim_matrix, l_matrix, is_balance=False, e=1e-8):

    _loss = -l_matrix * torch.log(sim_matrix + e) - \
            (1 - l_matrix) * torch.log(1 - sim_matrix + e)

    if is_balance:
        num_q, num_s = l_matrix.size(0), l_matrix.size(1)
        pos_idx = l_matrix == 1
        total_num, pos_num = num_q * num_s, torch.sum(pos_idx).item()
        _idx = torch.rand((num_s, num_q)) >= (pos_num / (total_num - pos_num))
        _idx[pos_idx] = False
        _loss[_idx] = 0

    _loss = torch.mean(_loss)
    return _loss


def proto_loss(similarity_matrix, label_matrix):
    label_matrix = torch.argmax(label_matrix, -1).long()
    return torch.nn.CrossEntropyLoss()(similarity_matrix, label_matrix)


l2_dis = euclidean_dis

# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     """
#
#     def __init__(self, margin=2.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#
#
#         return loss_contrastive
