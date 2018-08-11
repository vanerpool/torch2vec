import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class SkipW2V(nn.Module):

    def __init__(self, voc_size, embed_dim):
        """
        Class implementing W2V.

        Args
        int(voc_size): the number of different words within the vocabulary
        int(embed_dim): the size of the embedding dimension.

        Note: Weights are sampled from a random uniform distribution in the range [-.5, .5) and divided by the length of the
        embedding layer.
        """
        super(SkipW2V, self).__init__()
        self.embed_dim = embed_dim
        self.first_layer = nn.Embedding(voc_size, embed_dim)
        self.second_layer = nn.Embedding(voc_size, embed_dim)

        # Initialization
        self.first_layer.weight.data = self.first_layer.weight.data.uniform_(-.5, .5)/embed_dim
        self.second_layer.weight.data = self.second_layer.weight.data.uniform_(-.5, .5)/embed_dim

    def forward(self, batch):
        batch = torch.LongTensor(batch)

        # First column stores the input while the second and third stores the target and negative samples
        batch_v_i = self.first_layer(batch[:, 0])
        batch_v_j = self.second_layer(batch[:, 1])
        batch_v_neg = -self.second_layer(batch[:, 2:])

        # Batch dot products for positive batches
        positive_batch_scores = F.logsigmoid(torch.sum(
            batch_v_i*batch_v_j, dim=1))

        # Batch dot products for negative samples
        negative_batch_scores = torch.sum(
            F.logsigmoid(torch.sum(batch_v_i.view(len(batch), 1, self.embed_dim)*batch_v_neg, dim=2)),
            dim=1)

        loss = positive_batch_scores + negative_batch_scores
        return -torch.mean(loss)
