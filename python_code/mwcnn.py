import torch
from torch import nn
import ptwt

def dwt3d(x):
    # x.shape == (batch, channel, depth, height, width)
    xL = (x[:,:,0::2,:,:] + x[:,:,1::2,:,:])/2
    xH = (x[:,:,0::2,:,:] - x[:,:,1::2,:,:])/2

    xLL = (xL[:,:,:,0::2,:] + xL[:,:,:,1::2,:])/2
    xLH = (xL[:,:,:,0::2,:] - xL[:,:,:,1::2,:])/2
    xHL = (xH[:,:,:,0::2,:] + xH[:,:,:,1::2,:])/2
    xHH = (xH[:,:,:,0::2,:] - xH[:,:,:,1::2,:])/2

    xLLL = (xLL[:,:,:,:,0::2] + xLL[:,:,:,:,1::2])/2
    xLLH = (xLL[:,:,:,:,0::2] - xLL[:,:,:,:,1::2])/2
    xLHL = (xLH[:,:,:,:,0::2] + xLH[:,:,:,:,1::2])/2
    xLHH = (xLH[:,:,:,:,0::2] - xLH[:,:,:,:,1::2])/2
    xHLL = (xHL[:,:,:,:,0::2] + xHL[:,:,:,:,1::2])/2
    xHLH = (xHL[:,:,:,:,0::2] - xHL[:,:,:,:,1::2])/2
    xHHL = (xHH[:,:,:,:,0::2] + xHH[:,:,:,:,1::2])/2
    xHHH = (xHH[:,:,:,:,0::2] - xHH[:,:,:,:,1::2])/2

    res = torch.cat((xLLL, xLLH, xLHL, xLHH, xHLL, xHLH, xHHL, xHHH), 1)
    return res

def iwt3d(x):
    l = 2
    # x.shape == (batch, channel, depth, height, width)
    in_batch, in_channel, in_depth, in_height, in_width = x.size()
    device, dtype = x.device, x.dtype
    out_batch, out_channel, out_depth, out_height, out_width = in_batch, int(in_channel/(l**3)), l*in_depth, l*in_height, l*in_width
    
    xLLL, xLLH, xLHL, xLHH, xHLL, xHLH, xHHL, xHHH = [x[:, i*out_channel:(i+1)*out_channel] for i in range(0,8)]

    x_r = torch.zeros((out_batch, out_channel, out_depth, out_height, out_width)).to(dtype).to(device)
    x_r[:, :, 0::2, 0::2 , 0::2]=(xLLL+xLLH + xLHL+xLHH) + (xHLL+xHLH + xHHL+xHHH)
    x_r[:, :, 0::2, 0::2 , 1::2]=(xLLL-xLLH + xLHL-xLHH) + (xHLL-xHLH + xHHL-xHHH)
    x_r[:, :, 0::2, 1::2 , 0::2]=(xLLL+xLLH - xLHL+xLHH) + (xHLL+xHLH - xHHL+xHHH)
    x_r[:, :, 0::2, 1::2 , 1::2]=(xLLL-xLLH - xLHL-xLHH) + (xHLL-xHLH - xHHL-xHHH)
    x_r[:, :, 1::2, 0::2 , 0::2]=(xLLL+xLLH + xLHL+xLHH) - (xHLL+xHLH + xHHL+xHHH)
    x_r[:, :, 1::2, 0::2 , 1::2]=(xLLL-xLLH + xLHL-xLHH) - (xHLL-xHLH + xHHL-xHHH)
    x_r[:, :, 1::2, 1::2 , 0::2]=(xLLL+xLLH - xLHL+xLHH) - (xHLL+xHLH - xHHL+xHHH)
    x_r[:, :, 1::2, 1::2 , 1::2]=(xLLL-xLLH - xLHL-xLHH) - (xHLL-xHLH - xHHL-xHHH)

    return x_r

class DWT3d(nn.Module):
    def __init__(self):
        super(DWT3d, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt3d(x)

class IWT3d(nn.Module):
    def __init__(self):
        super(IWT3d, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt3d(x)

class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class MWCNN3d(nn.Module):
    def __init__(self, input_channels, out_channels, layer_num=1) -> None:
        super(MWCNN3d, self).__init__()
        self.dwt = DWT3d()
        self.iwt = IWT3d()
        feature = input_channels*8

        self.down = nn.ModuleList()
        for i in range(layer_num):
            self.down.append(DoubleConv3d(in_channels=feature, out_channels=feature))
            feature = feature*8

        self.bottom = DoubleConv3d(feature, feature)

        self.up = nn.ModuleList()
        for i in range(layer_num):
            feature = feature//8
            self.up.append(DoubleConv3d(feature, feature))
        
        self.final_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding='same')
    
    def forward(self, x):
        skip_conn = []
        
        # down
        x = self.dwt(x)
        for i, down in enumerate(self.down):
            x = down(x)
            skip_conn.append(x)
            x = self.dwt(x)

        # bottom
        x = self.bottom(x)

        # up
        skip_conn.reverse()
        for i, up in enumerate(self.up):
            x = self.iwt(x)
            skip = skip_conn[i]
            x = x + skip
            x = up(x)

        x = self.iwt(x)
        x = self.final_conv(x)

        return(x)