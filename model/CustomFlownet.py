from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import Contextnet,Unet

# RGB_channels
c=240

#Helper function for conv layers(from IFNet)
def conv(in_planes,out_planes,kernel_size=3,stride=1,padding=1,dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
                  nn.PReLU(out_planes)
    )

#Helper function for downsampling(from timelens unet)
class DownBlock(nn.Module):
    def __init__(self, inChannels,outChannels,filtersize):
        super(DownBlock,self).__init__()
        self.conv1 = nn.Conv2d(
            inChannels,
            outChannels,
            filtersize,
            stride = 1,
            padding=int((filtersize-1)/2),
        )
        self.conv2 = nn.Conv2d(
            outChannels,
            outChannels,
            filtersize,
            stride=1,
            padding=int((filtersize-1)/2),
        )

    def forward(self,x):
        x = F.avg_pool2d(x,2)
        x = F.leaky_relu(self.conv1(x),negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x),negative_slope=0.1)
        return x   

#VoxelEncoder: Based on timelens Unet encoder structure
class VoxelEncoder(nn.Module):
    def __init__(self,in_ch,out_ch=32):
        super().__init__()
        #Initial conv layers
        self.conv1 = nn.Conv2d(in_ch,32,7,stride=1,padding=3)
        self.conv2 = nn.Conv2d(32,32,7,stride=1,padding=3)
        
        #Downsampling path(encoder)
        self.down1 = DownBlock(32,64,5)
        self.down2 = DownBlock(64,128,3)
        self.down3 = DownBlock(128,256,3)
        self.down4 = DownBlock(256,512,3)
        self.down5 = DownBlock(512,512,3)

        #Projection to desired output channels
        self.proj = nn.Conv2d(512,out_ch,3,stride=1,padding=1)

    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x),negative_slope=0.1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        return x

#RGB Encoder:First half of IFBlock(conv0+convBlock)
class RGBEncoder(nn.Module):
    def __init__(self,in_planes = 6,c = 240) :
        super(RGBEncoder,self).__init__()
        #conv0:downsampling part
        self.conv0 = nn.Sequential(
            conv(in_planes,c//2,3,2,1),
            conv(c//2,c,3,2,1),
        )
        #convblock:feature processing with residual connection
        self.convbblock = nn.Sequential(
            conv(c,c),
            conv(c,c),
            conv(c,c),
            conv(c,c),
            conv(c,c),
            conv(c,c),
            conv(c,c),
            conv(c,c),
        )
    def forward(self,x):
        x = self.conv0(x)
        x = self.convbblock(x)+x
        return x

#Fusion module:Combines RGB and Voxel features
class FusionModule(nn.Module):
    def __init__(self,rgb_channels=240,voxel_channels=256,out_channels=240):
        super(FusionModule,self).__init__()
        # both features should have same spatial size
        # Voxel features are at 1/32 resolution, RGB features are at 1/4 resolution
        # upsample voxel features to match RGB features
        self.voxel_upsample = nn.Sequential(
            nn.ConvTranspose2d(voxel_channels,voxel_channels,4,stride=2,padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.ConvTranspose2d(voxel_channels,voxel_channels,4,stride=2,padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.ConvTranspose2d(voxel_channels,voxel_channels,4,stride=2,padding=1),
        )

        #Fusion convolution
        self.fusion_conv = nn.Sequential(
            conv(rgb_channels+voxel_channels,out_channels),
            conv(out_channels,out_channels),
        )
    def forward(self,rgb_feat,voxel_feat):
        #upsample voxel features to match with Rgb
        voxel_feat = self.voxel_upsample(voxel_feat)

        #ensure spatial match
        if voxel_feat.shape[2:] != rgb_feat.shape[2:]:
            voxel_feat = F.interpolate(
                voxel_feat,
                size = rgb_feat.shape[2:],
                mode = 'bilinear',
                align_corners=False
            ) 
        #Concat and fuse
        fused = torch.cat([rgb_feat,voxel_feat],dim = 1)
        fused = self.fusion_conv(fused)
        return fused

#Flow Decoder:Similar to IFBlock's lastconv,5 refers to 4(flow)+1(mask)
class FlowDecoder(nn.Module):
    def __init__(self,in_channels=240):
        super(FlowDecoder,self).__init__()
        self.lastconv = nn.ConvTranspose2d(in_channels,5,4,2,1)

    def forward(self,x,scale = 1):
        tmp = self.lastconv(x)
        #upsample to original resolution
        tmp = F.interpolate(tmp,scale_factor=scale*2,mode = 'bilinear',align_corners=False)
        flow = tmp[:,:4]*scale*2
        mask = tmp[:,4:5]
        return flow,mask
        

class CustomFlownet(nn.Module):
    def __init__(self,voxel_bins):
        super(CustomFlownet,self).__init__()
        #Encoders
        self.rgb_encoder = RGBEncoder(in_planes=6,c=c)
        self.voxel_encoder = VoxelEncoder(in_ch = voxel_bins,out_ch=256)

        #Fusion module
        self.fusion = FusionModule(rgb_channels=c,voxel_channels = 256,out_channels = c)

        #Flow decoder
        self.flow_decoder = FlowDecoder(in_channels=c)

        #Refinement net(same as IFNet)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self,x,voxel_grid,scale=[4,2,1],timestep=0.5):
         """
        Args:
            x: Input tensor [B, 6, H, W] (img0 + img1) or [B, 9, H, W] (with gt)
            voxel_grid: Voxel grid tensor [B, nb_of_time_bins, H, W]
            scale: List of scales for multi-scale processing
            timestep: Interpolation timestep (0.0 to 1.0)
        
        Returns:
            flow_list: List of flows at different scales [flow, flow, flow]
            mask: Final mask [B, 1, H, W]
            merged: List of merged images at different scales
            flow_teacher: None (removed teacher-student)
            merged_teacher: None (removed teacher-student)
            loss_distill: 0 (removed teacher-student)
        """
       
         img0 = x[:, :3]
         img1 = x[:,3:6]
         gt = x[:,6:]if x.shape[1] >6 else None
        # ============================================
        # REPLACE THIS SECTION WITH YOUR FLOW MODULE
        # ============================================
        # Your custom flow estimation should go here
        # Expected output: flow with shape [B, 4, H, W]
        # flow[:, :2] = flow from img0
        # flow[:, 2:4] = flow from img1
        
        # Example placeholder (replace with your actual flow computation):
        # flow = self.your_flow_network(img0, img1, timestep)
        
        # For now, this is a placeholder that returns zeros

         B,C,H,W = img0.shape
         #Encode RGB images
         rgb_input = torch.cat([img0,img1],dim=1) #[B,6,H,W]
         rgb_feat = self.rgb_encoder(rgb_input)   #[B,c,H/4,W/4]
         #Encode voxel grid
         voxel_feat = self.voxel_encoder(voxel_grid) #[B,256,H/32,W/32]
         #fuse
         fused_feat = self.fusion(rgb_feat,voxel_feat)

         #Decode flow and mask
         flow,mask = self.flow_decoder(fused_feat,scale=1)

         #warp images using flows
         warped_img0 = warp(img0,flow[:, :2])
         warped_img1 = warp(img1,flow[:, 2:4])

        #create mask
         mask = torch.sigmoid(mask)

        #create merged images
         merged_final = warped_img0 * mask + warped_img1 * (1-mask)

         # For compatibility, return in the same format as IFNet
         # flow_list: List of flows at different scales
         flow_list = [flow,flow,flow]

         #merged list: list of flows at different scales
         merged = [
            warped_img0 * mask + warped_img1 * (1-mask),
            warped_img0 * mask + warped_img1 * (1-mask),
            merged_final,
         ]

         #Teacher-student removed-always return none
         flow_teacher = None
         merged_teacher = None
         loss_distill = None

         #Refinement using contextnet and unet
         c0 = self.contextnet(img0,flow[:, :2])
         c1 = self.contextnet(img1,flow[:, 2:4])
         tmp = self.unet(img0,img1,warped_img0,warped_img1,mask,flow,c0,c1)
         res = tmp[:, :3] *2 -1
         merged_final = torch.clamp(merged_final + res,0,1)

         return flow_list,mask,merged,flow_teacher,merged_teacher,loss_distill