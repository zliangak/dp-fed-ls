import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """
    Linear model
    """

    def __init__(self):
        super(Linear, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        h = self.fc(x)
        return h


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNMnist(nn.Module):
    '''https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg.py'''
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1,  padding=2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class LSTM(nn.Module):
    def __init__(self,
                 input_size=80,
                 hidden_size=256,
                 embedding_dim=8,
                 num_classes=82,
                 n_layers=2
                 ):

        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        self.encoder = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True)
        self.decoder = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        #self.lstm.flatten_parameters()
        zx, hidden = self.lstm(x)
        output = self.decoder(zx[:, -1, :])
        return output


class CNNSvhn(nn.Module):
    def __init__(self, dropout=False):
        super(CNNSvhn, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv1.weight.data.normal_(std=1e-4)
        self.conv1.bias.data.fill_(0.0)

        self.pool1 = nn.MaxPool2d(3, 2, padding=1)

        self.norm1 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1)

        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv2.weight.data.normal_(std=1e-4)
        self.conv2.bias.data.fill_(0.1)

        self.norm2 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1)

        self.pool2 = nn.MaxPool2d(3, 2, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 384)
        self.fc1.weight.data.normal_(std=0.04)
        self.fc1.bias.data.fill_(0.1)

        self.fc2 = nn.Linear(384, 192)
        self.fc2.weight.data.normal_(std=0.04)
        self.fc2.bias.data.fill_(0.1)

        self.fc3 = nn.Linear(192, 10)
        self.fc3.weight.data.normal_(std=1 / 192.0)
        self.fc3.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm1(self.pool1(x))

        x = F.relu(self.conv2(x))
        x = self.pool2(self.norm2(x))

        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x
