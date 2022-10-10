import torch

class Perceptron(torch.nn.Module):
    """
    one Linear layer network
    """
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1)

    def forward(self, x_dim):
        return torch.sigmoid(self.fc1(x_dim))