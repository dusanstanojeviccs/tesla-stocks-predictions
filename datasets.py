import torch
import torch.utils.data.dataset as dataset

class LSTM_Stock_Dataset(dataset.Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        return torch.tensor(self.lines[idx], dtype=torch.float32)
    
class Stock_Dataset(dataset.Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        
        x = torch.tensor(self.lines[idx][:8], dtype=torch.float32)
        y = torch.tensor(self.lines[idx][8:], dtype=torch.float32)
        return {'x': x, 'y': y}