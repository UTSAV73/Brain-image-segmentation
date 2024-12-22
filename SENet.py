import torch
import torch.nn as nn

# SE Block implementation
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        # reduction ration of 8 is optimal as mentioned in the research paper
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        se = self.global_avg_pool(x).view(batch, channel) 
        se = self.fc(se).view(batch, channel, 1, 1)  # Fully connected Layer
        ###############Each channel of the feature map is scaled by its corresponding attention weight from se.#######################
        ####Channels with lower weights are suppressed(less relevant features).###
        return x * se  # Input map in scaled

class DoubleConv(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        down = self.conv(x)
        down = self.se(down)  #### Applied SE Block for down-sampling
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        x = self.conv(x)
        return self.se(x)  ###### Applied SE Block for up-sampling

# UNet with SE Blocks
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = nn.Sequential(
            DoubleConv(512, 1024),
            SEBlock(1024)  ######## Applied SE Block to bottleneck as well
        )

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out

# #Just a test code , output should be [1,1,256,256]
# if __name__ == "__main__":
#     input_image = torch.rand((1, 3, 256, 256)) 
#     model = UNet(3, 1)
#     output = model(input_image)
#     print(output.size()) 
