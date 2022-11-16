import torch.nn as nn
import torch.nn.functional as F
from modules.hgcn.layers import HGNN_conv

import torch.nn as nn
import torch.nn.functional as F


class HGCNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HGCNAgent, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.hgc1 = HGNN_conv(input_shape, args.hidden_dim)
        self.hgc2 = HGNN_conv(args.hidden_dim, args.n_feature)
        self.fc3 = nn.Linear(args.n_feature, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.hgc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    # todo inputs -> x, G (G通过H计算)
    def forward(self, x, G, hidden_state, actions=None):
        x = F.relu(self.hgc1(x, G))  # 激活函数Relu
        x = F.dropout(x, self.dropout)  # 使用dropout防止过拟合
        x = self.hgc2(x, G)
        actions = F.tanh(self.fc3(x))  # todo 全连接层输入？
        # x = self.hgc2(x, G)
        # actions = F.tanh(x)  # ? action?
        return {"actions": actions, "hidden_state": hidden_state}
