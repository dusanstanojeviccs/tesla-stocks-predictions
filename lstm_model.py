import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(4, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input.view(1, 1, -1), (hidden, cell))
        output = nn.functional.relu(self.fc1(output))
        output = nn.functional.relu(self.fc2(output))
        output = self.fc3(output)
        
        return output, (hidden, cell)
    
    def init_hidden(self):
        hidden = torch.zeros(1, 1, self.hidden_dim)
        cell = torch.zeros(1, 1, self.hidden_dim)

        return (hidden, cell)

    def evaluate(self, model, test_loader, hidden, cell, loss_function):
        loss = 0
        
        for i in range(len(test_loader) - 1):
            inputs = test_loader[i]
            labels = test_loader[i + 1]
            
            output, (hidden, cell) = model.forward(inputs, hidden, cell)

            loss += loss_function(output, labels).item()

        return loss
