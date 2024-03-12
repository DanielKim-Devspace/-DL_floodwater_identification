import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Encoder(nn.Module): #Unused
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
                
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1, 
            kernel_size=4, 
            stride=2)

        self.activation = nn.ReLU()
        
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        
        return x
    
    
class Decoder(nn.Module): #Unused
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upconv = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            padding=1, 
            kernel_size=4, 
            stride=2)
        
        self.activation = nn.ReLU()

        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        
        
    def forward(self, x, add_layer):
        
        x = self.upconv(x)
        x = x + add_layer
        x = self.activation(x)
        x = self.batch_norm(x)
        
        return x

class MyUnet(nn.Module):
    def __init__(self, num_blocks=4, in_channels=2, init_features=64):
        super(MyUnet, self).__init__()
        self.num_blocks = num_blocks
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final = nn.Conv2d(init_features, 2, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        
        j = 1
        for i in range(num_blocks):
            if i == 0:
                self.enc_blocks.append(MyUnet._block(in_channels, init_features, name='enc' + str(i)))
                self.dec_blocks.append(MyUnet._block(init_features * 2, init_features, name='dec' + str(i)))
                self.up_convs.append(nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2))
            else:
                self.enc_blocks.append(MyUnet._block(init_features * j, init_features * (j * 2), name='enc' + str(i)))
                self.dec_blocks.append(MyUnet._block(init_features * (j * 4), init_features * (j * 2), name='dec' + str(i)))
                self.up_convs.append(nn.ConvTranspose2d(init_features * (j * 4), init_features * (j * 2), kernel_size=2, stride=2))
                j *= 2
                
        self.bottleneck = MyUnet._block(init_features * j, init_features * (j * 2), name='bottleneck')

    def forward(self, x):
        encs = []
        decs = []
        
        for i in range(self.num_blocks):
            if i == 0:
                encs.append(self.enc_blocks[0](x))
            else:
                encs.append(self.enc_blocks[i](self.dropout(self.pool(encs[i-1]))))
        
        bottleneck = self.bottleneck(self.pool(encs[-1]))

        for i in range(self.num_blocks):
            if i == 0:
                decs.append(self.up_convs[-1](bottleneck))
                decs[-1] = torch.cat((decs[-1], encs[-1]), dim=1)
                decs[-1] = self.dec_blocks[-1](decs[-1])
            else:
                decs.append(self.up_convs[-1-i](decs[-1]))
                decs[-1] = torch.cat((decs[-1], encs[-1-i]), dim=1)
                decs[-1] = self.dec_blocks[-1-i](self.dropout(decs[-1]))

        scores = self.final(decs[-1])

        out = {
            'out': scores
        }
        return out

    # https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    
# Modification to standard Unet using Dropout2D layers.    
class MyUnet2(nn.Module):
    def __init__(self, num_blocks=4, in_channels=2, init_features=64, dropout_chance=.2):
        super(MyUnet2, self).__init__()
        self.num_blocks = num_blocks
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final = nn.Conv2d(init_features, 2, kernel_size=1)
        self.dropout_chance = dropout_chance
        
        
        j = 1
        for i in range(num_blocks):
            if i == 0:
                self.enc_blocks.append(MyUnet2._block2(in_channels, init_features, name='enc' + str(i), dropout_chance=self.dropout_chance))
                self.dec_blocks.append(MyUnet2._block2(init_features * 2, init_features, name='dec' + str(i), dropout_chance=self.dropout_chance))
                self.up_convs.append(nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2))
            else:
                self.enc_blocks.append(MyUnet2._block2(init_features * j, init_features * (j * 2), name='enc' + str(i), dropout_chance=self.dropout_chance))
                self.dec_blocks.append(MyUnet2._block2(init_features * (j * 4), init_features * (j * 2), name='dec' + str(i), dropout_chance=self.dropout_chance))
                self.up_convs.append(nn.ConvTranspose2d(init_features * (j * 4), init_features * (j * 2), kernel_size=2, stride=2))
                j *= 2
                
        self.bottleneck = MyUnet2._block2(init_features * j, init_features * (j * 2), name='bottleneck', dropout_chance=self.dropout_chance)

    def forward(self, x):
        encs = []
        decs = []
        
        for i in range(self.num_blocks):
            if i == 0:
                encs.append(self.enc_blocks[0](x))
            else:
                encs.append(self.enc_blocks[i](self.pool(encs[i-1])))
        
        bottleneck = self.bottleneck(self.pool(encs[-1]))

        for i in range(self.num_blocks):
            if i == 0:
                decs.append(self.up_convs[-1](bottleneck))
                decs[-1] = torch.cat((decs[-1], encs[-1]), dim=1)
                decs[-1] = self.dec_blocks[-1](decs[-1])
            else:
                decs.append(self.up_convs[-1-i](decs[-1]))
                decs[-1] = torch.cat((decs[-1], encs[-1-i]), dim=1)
                decs[-1] = self.dec_blocks[-1-i](decs[-1])

        scores = self.final(decs[-1])

        out = {
            'out': scores
        }
        return out

    # https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
    @staticmethod
    def _block2(in_channels, features, name, dropout_chance):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "drop1", nn.Dropout2d(p=dropout_chance, inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                            bias=True,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "drop2", nn.Dropout2d(p=dropout_chance, inplace=True))
                ]
            )
        )

# Danny / Simon Attention Unet
#https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""
   
    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):

    def __init__(self, img_ch=2, output_ch=2, dropout_chance=.1):
        super(AttentionUNet, self).__init__()
        #self.dropout_chance = dropout_chance
        self.dropout1 = nn.Dropout(p=dropout_chance)
        self.dropout2 = nn.Dropout(p=dropout_chance)
        self.dropout3 = nn.Dropout(p=dropout_chance)
        self.dropout4 = nn.Dropout(p=dropout_chance)
        self.dropout5 = nn.Dropout(p=dropout_chance)
        self.dropout6 = nn.Dropout(p=dropout_chance)
        self.dropout7 = nn.Dropout(p=dropout_chance)
        self.dropout8 = nn.Dropout(p=dropout_chance)
        self.dropout9 = nn.Dropout(p=dropout_chance)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.dropout1(self.Conv1(x))

        e2 = self.MaxPool(e1)
        e2 = self.dropout2(self.Conv2(e2))

        e3 = self.MaxPool(e2)
        e3 = self.dropout3(self.Conv3(e3))

        e4 = self.MaxPool(e3)
        e4 = self.dropout4(self.Conv4(e4))

        e5 = self.MaxPool(e4)
        e5 = self.dropout5(self.Conv5(e5))

        d5 = self.dropout6(self.Up5(e5))
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.dropout7(self.Up4(d5))
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.dropout8(self.Up3(d4))
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.dropout9(self.Up2(d3))
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)
        out = {
            'out': self.Conv(d2)
            }
        return out
    
    

    
    
    
    
#Modification to Attention UNet with Dropout2d layers and upsampling done within the attention block.
class ConvBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_chance):
        super(ConvBlock2, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_chance),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_chance)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv2, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock2(nn.Module):
    """Attention block with learnable parameters"""
   
    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock2, self).__init__()

        """self.W_gate = nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True)

        self.W_x = nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=2, padding=0, bias=True)"""

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.batchnorm = nn.BatchNorm2d(num_features=n_coefficients)


    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        
        psi = self.batchnorm(self.relu(g1 + x1)) #put a batchnorm here.
        psi = self.psi(psi)
        psi = self.upsample(psi)
        out = skip_connection * psi
        return out

class AttentionUNet2(nn.Module):

    def __init__(self, img_ch=2, output_ch=2, dropout_chance=.1):
        super(AttentionUNet2, self).__init__()
        self.dropout_chance = dropout_chance
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock2(img_ch, 64, dropout_chance = self.dropout_chance)
        self.Conv2 = ConvBlock2(64, 128, dropout_chance = self.dropout_chance)
        self.Conv3 = ConvBlock2(128, 256, dropout_chance = self.dropout_chance)
        self.Conv4 = ConvBlock2(256, 512, dropout_chance = self.dropout_chance)
        self.Conv5 = ConvBlock2(512, 1024, dropout_chance = self.dropout_chance)

        self.Up5 = UpConv2(1024, 512)
        self.Att5 = AttentionBlock2(F_g=1024, F_l=512, n_coefficients=512)
        self.UpConv5 = ConvBlock2(1024, 512, dropout_chance = self.dropout_chance)

        self.Up4 = UpConv2(512, 256)
        self.Att4 = AttentionBlock2(F_g=512, F_l=256, n_coefficients=256)
        self.UpConv4 = ConvBlock2(512, 256, dropout_chance = self.dropout_chance)

        self.Up3 = UpConv2(256, 128)
        self.Att3 = AttentionBlock2(F_g=256, F_l=128, n_coefficients=128)
        self.UpConv3 = ConvBlock2(256, 128, dropout_chance = self.dropout_chance)

        self.Up2 = UpConv2(128, 64)
        self.Att2 = AttentionBlock2(F_g=128, F_l=64, n_coefficients=64)
        self.UpConv2 = ConvBlock2(128, 64, dropout_chance = self.dropout_chance)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)
        
        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        
        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        
        d4 = self.Up5(e5)
        s4 = self.Att5(gate=e5, skip_connection=e4)
        d4 = torch.cat((s4, d4), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d4 = self.UpConv5(d4)

        d3 = self.Up4(d4)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d3 = torch.cat((s3, d3), dim=1)
        d3 = self.UpConv4(d3)

        d2 = self.Up3(d3)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d2 = torch.cat((s2, d2), dim=1)
        d2 = self.UpConv3(d2)

        d1 = self.Up2(d2)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d1 = torch.cat((s1, d1), dim=1)
        d1 = self.UpConv2(d1)
        out = {
            'out': self.Conv(d1)
            }
        return out
    
    
#linknet
#https://www.kaggle.com/code/mikhailsokolenko/lips-segmentation-linknet-pytorch/notebook
class EncoderLink(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
                
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=1, kernel_size=4, stride=2)

        self.activation = nn.ReLU()
        
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        
        return x
    
    
class DecoderLink(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, padding=1, kernel_size=4, stride=2)
        
        self.activation = nn.ReLU()

        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        
        
    def forward(self, x, add_layer):
        
        x = self.upconv(x)
        x = x + add_layer
        x = self.activation(x)
        x = self.batch_norm(x)
        
        return x

    
class LinkNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.a1 = EncoderLink(in_channels=2, out_channels=32)
        
        self.a2 = EncoderLink(in_channels=32, out_channels=64)
        
        self.a3 = EncoderLink(in_channels=64, out_channels=128)
        
        self.a4 = EncoderLink(in_channels=128, out_channels=256)
        
        self.a5 = EncoderLink(in_channels=256, out_channels=512)
        
        self.b = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=4, stride=2)
        
        self.c1 = DecoderLink(in_channels=512, out_channels=512)
        
        self.c2 = DecoderLink(in_channels=512, out_channels=256)
        
        self.c3 = DecoderLink(in_channels=256, out_channels=128)
        
        self.c4 = DecoderLink(in_channels=128, out_channels=64)
        
        self.c5 = DecoderLink(in_channels=64, out_channels=32)
        
        self.d = nn.ConvTranspose2d(in_channels=32, out_channels=2, padding=1, kernel_size=4, stride=2)
        
    def forward(self, x):
        
        a1 = self.a1(x)
        a2 = self.a2(a1)
        a3 = self.a3(a2)
        a4 = self.a4(a3)
        a5 = self.a5(a4)
        
        b = self.b(a5)
        b = nn.functional.relu(b)
        
        c1 = self.c1(b, a5)
        c2 = self.c2(c1, a4)
        c3 = self.c3(c2, a3)
        c4 = self.c4(c3, a2)
        c5 = self.c5(c4, a1)
        
        output = self.d(c5)
        output = nn.functional.sigmoid(output)
        
        out = {'out': output}
        
        return out 