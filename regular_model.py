import torch
from torch import nn

class RegularModel(nn.Module):
    def __init__(self):
        super(RegularModel, self).__init__()
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 4)

    def forward(self, input):
        output = nn.functional.relu(self.fc1(input))
        output = self.fc2(output)
        return output


    def evaluate(self, model, test_loader, loss_function):
        loss = 0

        for data in test_loader:
            inputs, labels = data['x'], data['y']
            output = model.forward(inputs)
            loss += loss_function(output, labels).item()

        return loss
