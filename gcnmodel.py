import torch
import torch.nn as nn
import torch.nn.functional as F

class gcn(nn.Module):

    def __init__(self, X_size, A_hat, bias=True):
        super(gcn, self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(X_size, 330))  # Hidden Size 1
        var = 2 / (self.weight.size(1) + self.weight.size(0))
        self.weight.data.normal_(0, var)
        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(330, 130))  # Hidden size 2
        var2 = 2 / (self.weight2.size(1) + self.weight2.size(0))
        self.weight2.data.normal_(0, var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(330))
            self.bias.data.normal_(0, var)
            self.bias1 = nn.parameter.Parameter(torch.FloatTensor(130))
            self.bias.data.normal_(0, var2)
        else:
            self.register_parameter()
        self.fc1 = nn.Linear(130, 6)  # Hidden size 2 and no of classes

    def forward(self, X):
        X = torch.mm(torch.FloatTensor(X), self.weight)
        if self.bias is not None:
            X = X + self.bias
        X = F.relu(torch.mm(self.A_hat, X))
        X = torch.mm(X, self.weight2)
        if self.bias is not None:
            X = X + self.bias1
        X = F.log_softmax(torch.mm(self.A_hat, X))

        return self.fc1(X)
