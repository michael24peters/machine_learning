import torch
import torch.nn.functional as F
from torch import nn

class ConvBNrelu(nn.Sequential):
    """convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super(ConvBNrelu, self).__init__(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(p)
#         Swish_module(),
)

class UNetPlusPlus(nn.Module):
    def __init__(self,
                 n=32,
                 sc_mode='concat',
                 dropout_p=0,
                 d_selection='ConvBNrelu',
                 u_selection='Up'
                ):
        super().__init__()
        if sc_mode == 'concat': 
            factor = 2
        else: 
            factor = 1
        self.mode = sc_mode
        self.p = dropout_p
        
        assert d_selection in downsample_options.keys(), f'Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}'
        assert u_selection in upsample_options.keys(), f'Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}'
        
        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]
                
        self.rcbn1 = d_block(1, n, kernel_size = 25, p=dropout_p)
        self.rcbn2 = d_block(n, n, kernel_size = 7, p=dropout_p)
        self.rcbn3 = d_block(n, n, kernel_size = 5, p=dropout_p)
        self.rcbn4 = d_block(n, n, kernel_size = 5, p=dropout_p)
        self.rcbn5 = d_block(n, n, kernel_size = 5, p=dropout_p)

        self.ui = nn.ConvTranspose1d(n, n, 2, 2)
        
        self.i1 = ConvBNrelu(2*n, n, kernel_size=5, p=dropout_p)
        self.i2 = ConvBNrelu(2*n, n, kernel_size=5, p=dropout_p)
        self.i3 = ConvBNrelu(2*n, n, kernel_size=5, p=dropout_p)
        self.i4 = ConvBNrelu(3*n, n, kernel_size=5, p=dropout_p)
        self.i5 = ConvBNrelu(3*n, n, kernel_size=5, p=dropout_p)
        self.i6 = ConvBNrelu(4*n, n, kernel_size=5, p=dropout_p)
        
        self.up1 = nn.ConvTranspose1d(n, n , 2, 2)
        self.up_c1 = ConvBNrelu(2*n, n, kernel_size=5, p=dropout_p)
        self.up2 = nn.ConvTranspose1d(n, n, 2, 2)
        self.up_c2 = ConvBNrelu(3*n, n, kernel_size=5, p=dropout_p)
        self.up3 = nn.ConvTranspose1d(n, n, 2, 2)
        self.up_c3 = ConvBNrelu(4*n, n, kernel_size=5, p=dropout_p)
        self.up4 = nn.ConvTranspose1d(n, n, 2, 2)
        self.up_c4 = ConvBNrelu(5*n, n, kernel_size=5, p=dropout_p)
        
        self.out_intermediate = nn.Conv1d(2*n, n, 5, padding=2) # padding=5-1//1
        self.outc = nn.Conv1d(n, 1, 5, padding=2) # padding=5-1//1
        
        self.d = nn.MaxPool1d(2)

    def forward(self, x):
#         x = torch.cat([x, x[:, ::-1, :]], dim=0) experimental - flip samples in a batch to try and  learn symmetrical kernels 
        
        d1 = self.rcbn1(x) # 4000
        d2 = self.d(self.rcbn2(d1)) # 2000
        d3 = self.d(self.rcbn3(d2)) # 1000
        d4 = self.d(self.rcbn4(d3)) # 500
        d5 = self.d(self.rcbn5(d4)) # 250
        
        ui1 = self.ui(d2)
        ui2 = self.ui(d3)
        ui3 = self.ui(d4)
        
        i1 = self.i1(torch.cat([ui1, d1], dim=1))
        i2 = self.i2(torch.cat([ui2, d2], dim=1))
        i3 = self.i3(torch.cat([ui3, d3], dim=1))
        
        ui4 = self.ui(i2)
        ui5 = self.ui(i3)
        
        i4 = self.i4(torch.cat([ui4, d1, i1], dim=1))
        i5 = self.i5(torch.cat([ui5, d2, i2], dim=1))
        
        ui6 = self.ui(i5)
        
        i6 = self.i6(torch.cat([ui6, d1, i1, i4], dim=1))
        
        u1 = self.up_c1(torch.cat([d4, self.up1(d5)], dim=1))             # 500
        u2 = self.up_c2(torch.cat([d3, i3, self.up2(u1)], dim=1))         # 1000
        u3 = self.up_c3(torch.cat([d2, i2, i5, self.up1(u2)], dim=1))     # 2000
        u4 = self.up_c4(torch.cat([d1, i1, i4, i6, self.up1(u3)], dim=1)) # 4000
        
        x = self.out_intermediate(torch.cat([u4, d1], dim=1))
        logits_x0 = self.outc(x)

        ret = F.softplus(logits_x0).squeeze()
        return  ret
    
    
    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    # If you are reading this, this function is probably not relevant to you. Carry on.
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNrelu:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                
