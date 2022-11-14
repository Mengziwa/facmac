import torch.nn as nn
import torch.nn.functional as F
from modules.hgcn.layers import HGNN_conv


class HGCNAgent(nn.Module):
    def __init__(self, input_shape, args, dropout=0.5):
        super(HGCNAgent, self).__init__()
        self.args = args
        self.dropout = dropout
        self.hgc1 = HGNN_conv(input_shape, args.hidden_dim)
        self.hgc2 = HGNN_conv(args.hidden_dim, args.n_actions)
        #self.f = nn.Linear(args.hidden_dim, args.n_actions)

        # self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    # def init_hidden(self):
    # make hidden states on same device as model
    #    return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))  # 激活函数Relu
        x = F.dropout(x, self.dropout)  # 使用dropout防止过拟合
        x = self.hgc2(x, G)
        actions = F.tanh(x) # ? action?
        return {"actions": actions, "feature": x}
