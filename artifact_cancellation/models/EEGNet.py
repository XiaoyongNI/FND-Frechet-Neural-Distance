import torch
import torch.nn as nn

class EEGNetModel(nn.Module): # EEGNET-8,2
    def __init__(self, chans=22, classes=4, time_points=1001, temp_kernel=32,
                 f1=8, f2=16, d=2, pk1=8, pk2=16, dropout_rate=0.5, max_norm1=1, max_norm2=0.25):
        super(EEGNetModel, self).__init__()
        # Calculating FC input features
        linear_size = (time_points//(pk1*pk2))*f2

        # Temporal Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        # Spatial Filters
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False), # Depthwise Conv
            nn.BatchNorm2d(d * f1),
            nn.ReLU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16),  groups=f2, bias=False, padding='same'), # Separable Conv
            nn.Conv2d(f2, f2, kernel_size=1, bias=False), # Pointwise Conv
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)
        self.mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)

        # Apply max_norm constraint to the linear layer
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        batch_size, channels, timesteps, features = x.shape
        # x = x.view(batch_size, channels, self.T,int(features/self.T))
        x = x.view(batch_size, 1,  channels, timesteps*features)
        # print(x[0])
        x_new = self.block1(x)
        # M = torch.rand(x_new.size()).float().to(self.mydevice) <= 0.5
        # print(x_new[0])
        # x_new = x_new*M + x
        x_new2 = self.block2(x_new)
        # print(x_new2[0])

        x_new3 = self.block3(x_new2)
        # print(x_new3[0])
        # input()
       
        x_new3 = self.flatten(x_new3)
        # x_new3 = self.fc(x_new3)
        
        return x_new3.squeeze()