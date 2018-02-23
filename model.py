import torch
from torch import nn
from torch.autograd import Variable


class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers, dropout, cuda):
        super().__init__()
        # model parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # model layers
        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, dropout)
        self.project = nn.Linear(hidden_size, num_classes)

        # use GPU?
        self.gpu = cuda

    def forward(self, x, hs=None):
        batch = x.shape[0]

        if hs is None:
            hs = Variable(torch.zeros(self.num_layers, batch, self.hidden_size))
            if self.gpu:
                hs = hs.cuda()

        word_embed = self.word_to_vec(x)  # (batch, len, embed)
        word_embed = word_embed.permute(1, 0, 2)  # (len, batch, embed)
        out, h_0 = self.rnn(word_embed, hs)  # (len, batch, hidden)
        length, num_batch, hidden_size = out.shape
        out = out.view(length * num_batch, hidden_size)
        out = self.project(out)
        out = out.view(length, num_batch, -1)
        out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden)

        return out.view(-1, out.shape[2]), h_0
