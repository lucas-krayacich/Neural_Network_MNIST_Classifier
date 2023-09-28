import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from torchsummary import summary

class autoencoderMLP4Layer(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = 392
        self.fc1 = nn.Linear(N_input,N2) #input = 1x784, output = 1x392
        self.fc2 = nn.Linear(N2, N_bottleneck) #output = 1xN
        self.fc3 = nn.Linear(N_bottleneck, N2) #output = 1x392
        self.fc4 = nn.Linear(N2, N_output) #output = 1x784
        self.type = 'MLP4'
        self.input_shape = (1, 28*28)

    def forward(self, x):
        #encoder
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        #decoder
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x

    def encode(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def decode(self, x):
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


#
if __name__ == '__main__':
    summary(autoencoderMLP4Layer(), (1, 28*28))

