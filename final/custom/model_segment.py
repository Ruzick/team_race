import torch
import torch.nn.functional as F
from torchvision import transforms
from . import dense_transforms

class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self,n_input_channels,n_output_channels,kernel_size=3, stride=2):#(self,n_input_channels,n_output_channels,kernel_size=3, stride=2):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input_channels,n_output_channels, kernel_size=3, stride=stride, padding=(kernel_size-1)//2, bias=False),
                torch.nn.BatchNorm2d(n_output_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                # torch.nn.MaxPool2d(kernel_size= 2* stride-1, stride=stride, padding=stride-1),
                # torch.nn.Conv2d(n_output_channels, n_output_channels, stride=2,kernel_size=3, padding=2, bias=False),
                torch.nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, padding=1, stride = 1,bias=False),
                torch.nn.BatchNorm2d(n_output_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2)
                
                )
            self.downsample = None
            if n_input_channels != n_output_channels or stride !=1 :
                self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(n_input_channels,n_output_channels, kernel_size=1, stride=stride ),
                torch.nn.BatchNorm2d(n_output_channels))

        def forward(self, x):
            ident = x
            if self.downsample is not None:
                ident = self.downsample(ident)
            return torch.nn.ReLU()(self.net(x) + ident) 
        
    def __init__(self, layers= [1000,100,150,70], n_input_channels=3):
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7, padding=3, stride=2, bias=False),
             torch.nn.BatchNorm2d(layers[0]),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]

        c = layers[0]

        for l in layers[1:]:
            L.append(self.Block(c, l))
            c = l

        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)
        torch.nn.init.zeros_(self.classifier.weight)


    def forward(self, x):
        transform_norm = transforms.Compose([
            #UTILS DENSE distribution
            transforms.Normalize([0.32352477, 0.3310059 , 0.34449455], [0.25328732, 0.22241966, 0.24833776])
            # transforms.ToTensor()
        ])
            # [0.02423741, 0.02405611 ,0.0258753 ] )#)[0.06797275, 0.05793738, 0.06476705])# [0.32351966 ,0.33099777 ,0.34447303],[0.02423741 ,0.02405611 ,0.0258753])# [0.02423741, 0.02405611 ,0.0258753 ] 
        x = transform_norm(x)
        y_pred = self.network(x)
        # Global average pooling
        y_pred = y_pred.mean(dim=[2,3])
        # Classify

        
        return self.classifier(y_pred)# ,per_channel_mean,per_channel_std )#[:,0]    
    # def __init__(self):
    #     super().__init__()
    #     """
    #     Your code here
    #     Hint: Base this on yours or HW2 master solution if you'd like.
    #     Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
    #     """
    #     raise NotImplementedError('CNNClassifier.__init__')

    # def forward(self, x):
    #     """
    #     Your code here
    #     @x: torch.Tensor((B,3,64,64))
    #     @return: torch.Tensor((B,6))
    #     Hint: Apply input normalization inside the network, to make sure it is applied in the grader
    #     """
    #     raise NotImplementedError('CNNClassifier.forward')



def Block(n_input_channels, n_output_channels):
    return torch.nn.Sequential(
                    # torch.nn.Conv2d(n_input_channels,n_output_channels,kernel_size=3, padding = 1),
                    #this can be factorized into 2 conv [ 1x3 with p=0 and 3x1 with p=1]
                    torch.nn.Conv2d(n_input_channels,n_output_channels,kernel_size=(1,3), padding = 0),
                    torch.nn.Conv2d(n_output_channels,n_output_channels,kernel_size=(3,1), padding = 1),
                    torch.nn.BatchNorm2d(n_output_channels),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1)
                    #too many parameters
                    # torch.nn.Conv2d(n_output_channels, n_output_channels,kernel_size=3, padding = 1),
                    # torch.nn.BatchNorm2d(n_output_channels),
                    # torch.nn.ReLU(),
                    # torch.nn.Dropout(0.1)
    )  
def BlockUp(n_input_channels, n_output_channels):
    return torch.nn.Sequential(
                    # torch.nn.Conv2d(n_input_channels,n_output_channels,kernel_size=3, padding = 1),
                    torch.nn.Conv2d(n_input_channels,n_output_channels,kernel_size=(1,3), padding = 0),
                    torch.nn.Conv2d(n_output_channels,n_output_channels,kernel_size=(3,1), padding = 1),
                    torch.nn.BatchNorm2d(n_output_channels),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1)
    )  



class FCN(torch.nn.Module):


    def __init__(self, layers= [32,64,128]):#,256]):
        super().__init__()
    #     """
    #     Your code here.
    #     Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
    #     Hint: Use up-convolutions
    #     Hint: Use skip connections
    #     Hint: Use residual connections
    #     Hint: Always pad by kernel_size / 2, use an odd kernel_size
    #     """
    
        #encoder
        #https://stackoverflow.com/questions/64780641/whats-the-equivalent-of-tf-keras-input-in-pytorch
        #https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
       
        self.conv1 = Block(3,layers[0]) # 3 to 32
        self.conv2 = Block(layers[0], layers[1]) #32 to 64
        self.conv3 = Block(layers[1], layers[2]) #64 to 128
        # self.conv4 = Block(layers[2], layers[3]) #128 to 256

        self.maxpool = torch.nn.MaxPool2d(kernel_size=4, padding=1, stride=2, dilation=1) #(H+2p-dil*(kern-1)-1)/str  +1  or dilation =3 kernel_size=2

        # self.upconv1 = torch.nn.ConvTranspose2d(layers[4], layers[3], kernel_size = 2, stride =2, padding = 0)
        # self.upconv3 = torch.nn.ConvTranspose2d(layers[3], layers[2], kernel_size = 3 ,stride =2, padding =1) #[32,64,128,256]
        # self.upconv2 = torch.nn.ConvTranspose2d(layers[2], layers[1], kernel_size = 3, stride =2, padding = 1)
        # self.upconv1 = torch.nn.ConvTranspose2d(layers[1], layers[0], kernel_size = 3, stride =2, padding =1)

        
        self.upscale = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        # self.up3 =BlockUp(layers[3], layers[2]) #256 to 128

        self.up2 = BlockUp(layers[2], layers[1]) #128 to 64
        self.up1 =BlockUp(layers[1], layers[0]) #64 to 32
        #layers= [32,64,128,256]
        self.up2cat = BlockUp(layers[2]*2, layers[1]) #128+128  to 64
        self.up1cat =BlockUp(layers[1]*2, layers[0]) #128  to 32

        # self.classifier = torch.nn.Conv2d(layers[0], 5, kernel_size=1)
        self.classifier = torch.nn.Conv2d(layers[0], 5, kernel_size=1)

    def forward(self, x):

        b, c, h, w = x.shape

        # max_dim = max(h,w)
        # #https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
        #https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        #https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
        class BobSquarePads:
            def __call__(self, image):
                max_wh = max(h,w)
                p_left, p_top = [(max_wh - s) // 2 for s in (h,w)]
                p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip((h,w), [p_left, p_top])]
                padding = (p_left, p_top, p_right, p_bottom)
                return F.pad(image, padding,'constant',0)

        target_image_size =(160,160)# (160,160)  # as an example images 128x96
        
        transform=transforms.Compose([
            # BobSquarePads(),
            transforms.Normalize([0.32352477, 0.3310059 , 0.34449455], [0.25328732, 0.22241966, 0.24833776]),
            transforms.Pad((128,128,128,128),padding_mode= 'constant',fill=0),
            transforms.CenterCrop(target_image_size)
         ])
        
        #************************************************************************************
        #          ENCODER                 3x128x96           3x128x128          3x80x80                                 3x160x160
        #*******************************************************************************
        x = transform(x)  #             3x128x96             3x128x128             3x80x80                                #3x160x160
        #first convolution
        conv1 =  self.conv1(x)  #    32x128x96              32x128x128           32x80x80                               32x160x160
        x= self.maxpool(conv1)  #   32x64x48              32x64x64               32x40x40                              32x80x80 
        #2ndconvolution
        conv2 =  self.conv2(x)     #  64x64x48               64x64x64             64x40x40                              64x80x80 
        x= self.maxpool(conv2)   # 64x32x24               64x32x32             64x20x20                              64x40x40

        #3rdconvolution
        conv3 =  self.conv3(x)     # 128x32x24            128x32x32            128x20x20                          #128x40x40
        x= self.maxpool(conv3)  #  128x16x12             128x16x16           128x10x10                            128x20x20

        #****************************************************************************************
        #           DECODER 1 (USING SKIPPED CONNECTIONS (Addition) +  UPSCALE) 128x16x16 
        #*******************************************************************************************

        #upconvolution 1
        x=  self.upscale(x)  #  128x32x24                  128x32x32           128x20x20              128x20x20 torch.Size([128, 128, 20, 20])
        x = x+ conv3
        x = self.up2(x) #     64x32x24                           64x32x32               64x20x20                    128x10x10  torch.Size([128, 128, 10, 10])


        #upconvolution2
        x=  self.upscale(x) #  64x64x48                       64x64x64               64x40x40             128x40x40     torch.Size([128, 128, 40, 40])
        x = x+conv2 #      x+conv3      torch.cat([x, conv3], dim=1)   #128x40x40   if concat then torch.Size([128, 256, 40, 40]) goes to 256 torch.Size([128, 256, 40, 40])
        x= self.up1(x) #  32x64x48                                        32x64x64                 32x40x40               64x40x40
        
        #upconvolution3

        x= self.upscale(x) #   32x128x96                                     32x 128x128                32x80x80           64x80x80
        x = x+conv1  # x = x+conv2


        #********************************************************************************************
        #             DECODER  2 (USING concatanations+  UPSCALE) 128x16x16 
        #************************************************************************************************

        # #upconvolution 1
        # x=  self.upscale(x)  #128x32x32           128x20x20              128x20x20 torch.Size([128, 128, 20, 20])
        # x = torch.cat([x, conv3], dim=1)   #258x32x32
        # x = self.up2cat(x) #       64x32x32               64x20x20                    128x10x10  torch.Size([128, 128, 10, 10])

        # #upconvolution2
        # x=  self.upscale(x) # 64x64x64               64x40x40             128x40x40     torch.Size([128, 128, 40, 40])
        # x = torch.cat([x, conv2], dim=1) #  128x64x64
        # x= self.up1cat(x) #      32x64x64                 32x40x40               64x40x40
        
        # #upconvolution3
        # x= self.upscale(x) # 32x 128x128                32x80x80           64x80x80
        # x = x+conv1
       


        #************************************************************************************
        #                   DECODER 3    128x16x12     128x16x16   ConvTranspose 2d
        #*************************************************************************************

        # x =torch.nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)(x) # 128x32x24

        # x = torch.cat([x, conv3], dim=1) #256 x 32x24

        # x = torch.nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)(x) #64x64x48

        # x = x+conv2

        # x =  torch.nn.ConvTranspose2d( 64, 32, kernel_size=2, stride=2)(x) #32x128x96

        # x = x+conv1

        #***************************************************************
        #                               c l a s i f i c a t i o n
        #***************************************************************
        y_pred = self.classifier(x)

        #resize after transform
        y_pred=transforms.CenterCrop((h,w))(y_pred)
        return y_pred

    # def forward(self, x):

    #     transform_norm = transforms.Normalize([0.32352477, 0.3310059 , 0.34449455], [0.25328732, 0.22241966, 0.24833776])
    #     x = transform_norm(x)
    #     y_pred = self.network(x)
    #     # print((y_pred [:, :, :h, :w]).shape)
    #     print(y_pred.shape)
    #     return self.classifier(y_pred)#[:, :, :h, :w]


        # """
        # Your code here
        # @x: torch.Tensor((B,3,H,W))
        # @return: torch.Tensor((B,5,H,W))
        # Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        # Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
        #       if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
        #       convolution
        # """
        # raise NotImplementedError('FCN.forward')


model_factory = {
    'fcn': FCN
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r


def cross_entropy2d(input, target, weight=None, size_average=True):
    #https://github.com/wkentaro/pytorch-fcn/blob/main/torchfcn/trainer.py
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)

    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

