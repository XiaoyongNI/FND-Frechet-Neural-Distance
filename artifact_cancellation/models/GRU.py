import torch
import torch.nn as nn
from torch.autograd import Variable

class MultiChannelGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device,T,channels, dropout=0.5):
        super(MultiChannelGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.h0 = torch.randn(num_layers, batch_size, hidden_size)  # 初始隐藏状态
        self.T = T   # 初始隐藏状态
        self.device = device
        self.channels = channels

    
    def forward(self, x):
        # x shape: [batch_size, channels, timesteps, features]
        if x.dim() == 3:
            x = x.view(-1, self.channels, x.shape[1], x.shape[2])
        # 将数据展平为 [batch_size * channels, timesteps, features] 进行处理
        batch_size, channels, timesteps, features = x.size()
        x = x.view(-1, timesteps, features)
        # x = (x - x.mean(2, keepdim=True))/(x.std(2, keepdim=True) +1e-10)

        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device))

        h_n = h0.to(self.device)
        # 通过 GRU 处理
        for t in range(self.T):
            seq = x[:,t,:].float().view(x.size(0), 1, x.size(2))
            out, h_n = self.gru(seq, h_n)

        out = self.fc(out)

        # 将输出恢复成 [batch_size, channels, output_size]
        out = out.view(batch_size, channels, -1).squeeze(dim=2)
        out = torch.mean(out, dim=1)

        return out 



class CombinedChannelGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device,T):
        super(CombinedChannelGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.T = T  
        self.device = device
    
    def forward(self, x):
        # x shape: [batch_size, channels, timesteps, features]
        # 将数据展平为 [batch_size * channels, timesteps, features] 进行处理
        batch_size, channels, timesteps, features = x.shape
        # x = x.view(batch_size, channels, self.T,int(features/self.T))
        x = x.view(batch_size, timesteps, channels * features)
        # x = (x - x.mean(2, keepdim=True))/(x.std(2, keepdim=True) +1e-10)


        if torch.cuda.is_available():
            h0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device))
        else:
            h0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size))

        lisths = []
        # 通过 GRU 处理
        h_n = h0

        for t in range(self.T):
            seq = x[:,t,:].float().view(x.size(0), 1, x.size(2))
            out, h_n = self.gru(seq, h_n)
            lisths.append(h_n)

        # # 取每个通道的最后时刻的输出
        # out = lisths[:, -1, :].squeeze() #batch,hidden_size
        
        # 输出层
        out = self.fc(out)
        
        return out 