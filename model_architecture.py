import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_family_classes):
        """
        Simple MLP model for toxin family classification

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            num_family_classes: Number of output classes
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_family = nn.Linear(hidden_dim, num_family_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc_family(x)
