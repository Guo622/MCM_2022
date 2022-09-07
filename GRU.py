import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


def seed_torch(seed=2022):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


def dataloader(df: pd.DataFrame, batch_size, num_steps, pre_steps):
    data = df[df.columns[1]].values
    m = data.mean()
    std = data.std()
    data = (data-m)/std
    data_len = len(data)
    num_examples = (data_len-1)//num_steps
    epoch_size = num_examples//batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return data[pos:pos+num_steps]

    def _data_(pos):
        return data[pos:pos+pre_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i:i+batch_size]
        X = np.array([_data(j*num_steps) for j in batch_indices])
        Y = np.array([_data_(j*num_steps+num_steps) for j in batch_indices])
        X = torch.from_numpy(X).float().cuda()
        Y = torch.from_numpy(Y).float().cuda()
        yield X.view(batch_size, num_steps, -1), Y


if __name__ == '__main__':
    net = GRU(input_size=1, hidden_size=32,
              num_layers=1, num_classes=3).to(device)
    df = pd.read_csv('data/BCHAIN-MKPRU.csv')
    epochs = 2000
    batch_size = 2
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    criterion = nn.MSELoss().to(device)
    net.train()
    losses = []
    for e in range(epochs):
        net.train()
        total_train_loss = 0
        dl = dataloader(df, batch_size=2, num_steps=7, pre_steps=3)
        for i, (x, y) in enumerate(dl):
            y_pred = net(x)
            loss = criterion(y_pred, y)
            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch[{e+1}/{epochs}] loss:{total_train_loss}')
        losses.append(total_train_loss)

    torch.save(net, 'gru_bit.pth')
