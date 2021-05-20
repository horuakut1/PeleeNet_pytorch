import torch
import torch.nn as nn
import torchsummary
# torch version is 1.2.0
class con(nn.Module):
    'conv_bn_relu,the basic block'
    def __init__(self,inplanes,planes,k,s=1,p=0,relu = True) -> None:
        super().__init__()
        if relu:
            self.con = nn.Sequential(
                nn.Conv2d(in_channels=inplanes,out_channels=planes,kernel_size=k,stride=s,padding=p),
                nn.BatchNorm2d(planes),
                nn.ReLU()
            )
        else:
            self.con = nn.Sequential(
                nn.Conv2d(in_channels=inplanes,out_channels=planes,kernel_size=k,stride=s,padding=p),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        x = self.con(x)
        return x

class DenseLayer(nn.Module):
    'the DenseLayer in paper'
    def __init__(self,inplanes,planes1,planes2) -> None:
        super().__init__()
        self.route1 = nn.Sequential(
            con(inplanes,planes1,k=1),
            con(planes1,planes2,k=3,p=1)
        )
        self.route2 = nn.Sequential(
            con(inplanes,planes1,k=1),
            con(planes1,planes2,k=3,p=1),
            con(planes2,planes2,k=3,p=1)
        )

    def forward(self, x):
        x1 = self.route1(x)
        x2 = self.route2(x)
        return torch.cat([x,x1,x2],dim=1)

class stem(nn.Module):
    'the first stage in the PeleeNet'
    def __init__(self,inplanes) -> None:
        super().__init__()
        self.con = con(inplanes,32,k=3,s=2,p=1)
        self.route1 = nn.Sequential(
            con(32,16,k=1),
            con(16,32,k=3,s=2,p=1)
        )
        self.route2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.con2 = nn.Sequential(
            con(64,32,k=1)
        )
    def forward(self,x):
        x = self.con(x)
        x1 = self.route1(x)
        x = self.route2(x)
        x = torch.cat([x1,x],dim=1)
        x = self.con2(x)
        return x

class TransLayer(nn.Module):
    'the layer between stages'
    def __init__(self,inplanes,planes):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,planes,kernel_size=1),
            nn.AvgPool2d(kernel_size=2,stride=2)
        )
    def forward(self, x):
        x = self.layer(x)
        return x

class PeleeNet(nn.Module):
    'the main net'
    def __init__(self) -> None:
        super().__init__()
        self.stem =stem(3) # this is the number of the input channel, you can change this number for your datasets
        self.stage1 = nn.Sequential(
            DenseLayer(32,16,16),
            DenseLayer(64,16,16),
            DenseLayer(96,16,16),
            TransLayer(128,128)
        )
        self.stage2 = nn.Sequential(
            DenseLayer(128,32,16),
            DenseLayer(160,32,16),
            DenseLayer(192,32,16),
            DenseLayer(224,32,16),
            TransLayer(256,256)
        )
        self.stage3 = nn.Sequential(
            DenseLayer(256,64,16),
            DenseLayer(288,64,16),
            DenseLayer(320,64,16),
            DenseLayer(352,64,16),
            DenseLayer(384,64,16),
            DenseLayer(416,64,16),
            DenseLayer(448,64,16),
            DenseLayer(480,64,16),
            TransLayer(512,512)
        )
        self.stage4 = nn.Sequential(
            DenseLayer(512,64,16),
            DenseLayer(544,64,16),
            DenseLayer(576,64,16),
            DenseLayer(608,64,16),
            DenseLayer(640,64,16),
            DenseLayer(672,64,16),
            TransLayer(704,704)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(704,10) # 10 is the number of the classes, you can change this number for your datasets

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = PeleeNet()
    print(model)
    # if you have GPU, You can use the following code to show this net
    # device = torch.device('cuda:0')
    # model.to(device=device)
    # torchsummary.summary(model,input_size=(3,416,416))

